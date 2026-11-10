"""Admin API and client authentication."""

from __future__ import annotations

import pytest

from tests.conftest import HOPLITE_KEY

pytestmark = pytest.mark.anyio


async def test_create_list_get_key(client, admin_headers) -> None:
    resp = await client.post("/admin/keys", headers=admin_headers, json={
        "hoplite_api_key": HOPLITE_KEY, "name": "ci", "default_model": "claude-sonnet-5",
        "spend_limit_micros": 250000, "max_concurrency": 2,
        "repos": [{"repo_full_name": "acme/app", "branch": "main"}],
    })
    assert resp.status_code == 201, resp.text
    created = resp.json()
    assert created["key"].startswith("sk-hop-")
    assert created["project_id"] == "proj_1"  # the only project is picked automatically
    assert created["hoplite_key_hint"] == "hop_…_key"
    assert created["repos"] == [{"repo_full_name": "acme/app", "branch": "main"}]

    listing = (await client.get("/admin/keys", headers=admin_headers)).json()
    assert [k["id"] for k in listing["data"]] == [created["id"]]
    entry = listing["data"][0]
    assert "key" not in entry and "secret_hash" not in entry and "hoplite_key_enc" not in entry
    assert entry["key_prefix"] == f"sk-hop-{created['id']}…"

    single = await client.get(f"/admin/keys/{created['id']}", headers=admin_headers)
    assert single.json()["spend_limit_micros"] == 250000


async def test_admin_auth_required(client, admin_headers) -> None:
    assert (await client.get("/admin/keys")).status_code == 401
    wrong = await client.get("/admin/keys", headers={"Authorization": "Bearer " + "x" * 40})
    assert wrong.status_code == 401 and wrong.json()["error"]["code"] == "invalid_admin_key"
    alt = await client.get("/admin/keys", headers={"X-Admin-Key": admin_headers["Authorization"][7:]})
    assert alt.status_code == 200


@pytest.mark.parametrize("settings_overrides", [{"master_key": "too-short"}])
async def test_admin_disabled_without_strong_master_key(client) -> None:
    resp = await client.get("/admin/keys", headers={"Authorization": "Bearer too-short"})
    assert resp.status_code == 503 and resp.json()["error"]["code"] == "admin_disabled"


async def test_upstream_validation(client, admin_headers, fake) -> None:
    bad_key = await client.post("/admin/keys", headers=admin_headers, json={"hoplite_api_key": "hop_unknown_key"})
    assert bad_key.status_code == 400 and bad_key.json()["error"]["code"] == "invalid_hoplite_key"

    not_hop = await client.post("/admin/keys", headers=admin_headers, json={"hoplite_api_key": "sk-something-else"})
    assert not_hop.status_code == 400

    missing = await client.post("/admin/keys", headers=admin_headers,
                                json={"hoplite_api_key": HOPLITE_KEY, "project_id": "proj_nope"})
    assert missing.status_code == 400 and missing.json()["error"]["code"] == "project_not_found"

    fake.projects.append({"id": "proj_2", "name": "Second"})
    ambiguous = await client.post("/admin/keys", headers=admin_headers, json={"hoplite_api_key": HOPLITE_KEY})
    assert ambiguous.status_code == 400 and ambiguous.json()["error"]["code"] == "project_required"
    assert "proj_2" in ambiguous.json()["error"]["message"]

    unchecked = await client.post("/admin/keys", headers=admin_headers,
                                  json={"hoplite_api_key": "hop_unverified", "project_id": "proj_x", "verify": False})
    assert unchecked.status_code == 201


async def test_client_auth_and_revocation(client, admin_headers, api_key) -> None:
    ok = await client.get("/v1/models", headers={"Authorization": f"Bearer {api_key}"})
    assert ok.status_code == 200

    for headers in ({}, {"Authorization": "Bearer nope"}, {"Authorization": f"Bearer {api_key[:-1]}X"}):
        resp = await client.get("/v1/models", headers=headers)
        assert resp.status_code == 401, headers
        assert resp.json()["error"]["code"] == "invalid_api_key"

    revoked = await client.delete(f"/admin/keys/{api_key}", headers=admin_headers)  # full key accepted, as in the spec
    assert revoked.status_code == 200 and revoked.json()["revoked"] is True
    after = await client.get("/v1/models", headers={"Authorization": f"Bearer {api_key}"})
    assert after.status_code == 401
    again = await client.delete(f"/admin/keys/{revoked.json()['id']}", headers=admin_headers)
    assert again.status_code == 404


async def test_patch_rotates_upstream_key(client, admin_headers, api_key, fake) -> None:
    key_id = api_key[len("sk-hop-"):][:12]
    resp = await client.patch(f"/admin/keys/{key_id}", headers=admin_headers,
                              json={"hoplite_api_key": "hop_second_key", "name": "rotated", "spend_limit_micros": 0})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "rotated" and body["hoplite_key_hint"] == "hop_…_key" and body["spend_limit_micros"] == 0

    fake.api_keys.discard(HOPLITE_KEY)  # the old upstream key no longer works
    models = await client.get("/v1/models", headers={"Authorization": f"Bearer {api_key}"})
    assert models.status_code == 200

    bad = await client.patch(f"/admin/keys/{key_id}", headers=admin_headers, json={"project_id": None})
    assert bad.status_code == 400
