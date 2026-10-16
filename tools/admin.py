"""
Manage teams from the command line, straight on the database (no server or admin key needed).
Same validation and admin log as the admin API; a running server applies changes on the next request.

    teams | show TEAM | new-team NAME | key TEAM
    mode TEAM observe|enforce | on TEAM FEATURE... | off TEAM FEATURE...
    set TEAM NAME=VALUE...          e.g. budget_daily_eur=50

    .venv/bin/python -m tools.admin mode "Support team" enforce
"""

import argparse
import os
import secrets
import sys
from pathlib import Path

import yaml

from src.paths import CONFIG_PATH, DATA_DIR
from src.protection import security
from src.server import settings as settings_module
from src.server.store import Store

DEFAULT_DB = DATA_DIR / "cordon.sqlite"


class AdminError(Exception):
    """A problem to report to the user in one line (unknown team, invalid value, ...)."""


# --- Section: helpers ---
def open_store(db_path: str | None) -> Store:
    """Open the database: --db, else CORDON_DB_PATH, else data/cordon.sqlite."""
    path = db_path or os.environ.get("CORDON_DB_PATH") or str(DEFAULT_DB)
    if not Path(path).exists():
        raise AdminError(f"no database at {path} - start Cordon once, or pass --db")
    return Store(path)


def find_team(store: Store, team: str) -> dict:
    """Return the tenant row whose id or name (case-insensitive) matches, or raise AdminError."""
    tenants = store.list_tenants()
    matches = [t for t in tenants if t["id"] == team] or [t for t in tenants if t["name"].lower() == team.lower()]
    if not matches:
        names = ", ".join(f'"{t["name"]}"' for t in tenants) or "none yet"
        raise AdminError(f'no team "{team}" (teams: {names})')
    if len(matches) > 1:
        raise AdminError(f'several teams are called "{team}" - use the id instead')
    return matches[0]


def apply_changes(store: Store, tenant: dict, overrides: dict, action: str) -> dict:
    """Validate overrides like the admin API does, save them, log them, and return the new settings."""
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        merged = settings_module.merge_tenant_settings(tenant["settings"], overrides, config)
    except ValueError as exc:
        raise AdminError(str(exc)) from None
    store.patch_tenant(tenant["id"], merged)
    store.write_admin_log(action, tenant["id"], f"fields={sorted(overrides)}")
    return merged


def summary_line(tenant: dict) -> str:
    """One line per team: name, id, mode and features."""
    s = tenant["settings"]
    features = ", ".join(s.get("features") or []) or "none"
    return f'{tenant["name"]:<24} id={tenant["id"]:<14} mode={s.get("mode"):<8} features: {features}'


def parse_assignments(pairs: list[str]) -> dict:
    """Turn ["budget_daily_eur=50", "pool=[gpt-4o-mini, gpt-4o]"] into typed values (YAML rules: 50 -> int, true -> bool)."""
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise AdminError(f'"{pair}" is not NAME=VALUE')
        name, raw = pair.split("=", 1)
        overrides[name.strip()] = yaml.safe_load(raw) if raw.strip() else None
    return overrides


# --- Section: commands ---
def cmd_teams(store: Store, args) -> None:
    """List all teams."""
    tenants = store.list_tenants()
    if not tenants:
        print("No teams yet. Create one with: new-team NAME")
    for tenant in tenants:
        print(summary_line(tenant))


def cmd_show(store: Store, args) -> None:
    """Print every setting of one team."""
    tenant = find_team(store, args.team)
    print(summary_line(tenant))
    for name, value in sorted(tenant["settings"].items()):
        print(f"  {name} = {value}")


def cmd_new_team(store: Store, args) -> None:
    """Create a team from config.yaml's secure defaults."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    tenant_id = secrets.token_urlsafe(9)
    store.create_tenant(tenant_id, args.name, settings_module.default_tenant_settings(config))
    store.write_admin_log("create_tenant", tenant_id, f"name={args.name}")
    print(f'Created team "{args.name}" (id {tenant_id}) in observe mode with no features on.')
    print(f'Next: give it a key with   key "{args.name}"')


def cmd_key(store: Store, args) -> None:
    """Issue a new API key for a team; the key is shown only now."""
    tenant = find_team(store, args.team)
    plaintext, key_hash = security.generate_api_key()
    prefix = security.key_prefix(plaintext)
    store.add_api_key(key_hash, prefix, tenant["id"], None)
    store.write_admin_log("create_api_key", tenant["id"], f"prefix={prefix}")
    print(f'New key for "{tenant["name"]}" (copy it now - it is never shown again):')
    print(plaintext)


def cmd_mode(store: Store, args) -> None:
    """Switch a team between observe and enforce."""
    tenant = find_team(store, args.team)
    apply_changes(store, tenant, {"mode": args.mode}, "patch_tenant")
    print(f'"{tenant["name"]}" is now in {args.mode} mode.')


def cmd_features(store: Store, args) -> None:
    """Switch features on or off."""
    tenant = find_team(store, args.team)
    # Only features that still exist: a team stored before one was retired must stay editable.
    current = [f for f in tenant["settings"].get("features") or [] if f in settings_module.KNOWN_FEATURES]
    if args.command == "on":
        updated = current + [f for f in args.features if f not in current]
    else:
        updated = [f for f in current if f not in args.features]
    merged = apply_changes(store, tenant, {"features": updated}, "patch_tenant")
    print(f'"{tenant["name"]}" features: {", ".join(merged["features"]) or "none"}')


def cmd_set(store: Store, args) -> None:
    """Change any setting(s)."""
    tenant = find_team(store, args.team)
    overrides = parse_assignments(args.assignments)
    merged = apply_changes(store, tenant, overrides, "patch_tenant")
    for name in overrides:
        print(f'"{tenant["name"]}": {name} = {merged[name]}')


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (see the module docstring)."""
    parser = argparse.ArgumentParser(prog="python -m tools.admin", description="Manage Cordon teams.")
    parser.add_argument("--db", help="database file (default: CORDON_DB_PATH or data/cordon.sqlite)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("teams", help="list all teams").set_defaults(func=cmd_teams)
    p = sub.add_parser("show", help="all settings of one team"); p.add_argument("team"); p.set_defaults(func=cmd_show)
    p = sub.add_parser("new-team", help="create a team"); p.add_argument("name"); p.set_defaults(func=cmd_new_team)
    p = sub.add_parser("key", help="give out a new API key"); p.add_argument("team"); p.set_defaults(func=cmd_key)
    p = sub.add_parser("mode", help="switch the mode"); p.add_argument("team")
    p.add_argument("mode", choices=["observe", "enforce"]); p.set_defaults(func=cmd_mode)
    for name, text in (("on", "switch features on"), ("off", "switch features off")):
        p = sub.add_parser(name, help=f"{text} ({', '.join(settings_module.KNOWN_FEATURES)})")
        p.add_argument("team"); p.add_argument("features", nargs="+"); p.set_defaults(func=cmd_features)
    p = sub.add_parser("set", help="change settings, e.g. budget_daily_eur=50"); p.add_argument("team")
    p.add_argument("assignments", nargs="+"); p.set_defaults(func=cmd_set)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on a reported problem."""
    args = build_parser().parse_args(argv)
    try:
        store = open_store(args.db)
        try:
            args.func(store, args)
        finally:
            store.close()
    except AdminError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
