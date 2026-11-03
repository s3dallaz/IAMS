"""
Account recovery from the command line, for when nobody can sign in (e.g.
the only IT account forgot its password). It needs access to this PC and
its database, which is why it does not ask for an account.

    .venv\\Scripts\\python scripts\\manage_accounts.py list
    .venv\\Scripts\\python scripts\\manage_accounts.py reset-password <username>
    .venv\\Scripts\\python scripts\\manage_accounts.py unlock <username>
    .venv\\Scripts\\python scripts\\manage_accounts.py create-it <username> "<full name>"

Passwords are typed in hidden, never passed on the command line.
"""

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.schema import initialize_database  # noqa: E402
from src.services.auth_service import (  # noqa: E402
    ROLE_LABELS, AuthError, AuthService, Role, hash_password, validate_password,
)


def _ask_password(username: str) -> str:
    password = getpass.getpass("New password: ")
    if password != getpass.getpass("Repeat it: "):
        raise SystemExit("The two passwords are different. Nothing changed.")
    try:
        validate_password(password, username)
    except AuthError as e:
        raise SystemExit(f"{e} Nothing changed.")
    return password


def _find(auth: AuthService, username: str) -> dict:
    user = auth.repository.get_by_username(username)
    if user is None:
        raise SystemExit(f"No account '{username}'. Use 'list' to see the accounts.")
    return user


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Account recovery")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show the accounts")
    for name in ("reset-password", "unlock"):
        commands.add_parser(name).add_argument("username")
    create = commands.add_parser("create-it", help="create an IT account")
    create.add_argument("username")
    create.add_argument("full_name")
    args = parser.parse_args(argv)

    initialize_database()
    auth = AuthService()
    if args.command == "list":
        for user in auth.repository.list_users():
            state = "active" if user['is_active'] else "deactivated"
            print(f"{user['username']:<30} {ROLE_LABELS[user['role']]:<11} {state}"
                  + (f", locked until {user['locked_until']} UTC" if user['locked_until'] else ""))
    elif args.command == "reset-password":
        user = _find(auth, args.username)
        auth.repository.set_password(user['id'], hash_password(_ask_password(user['username'])),
                                     must_change_password=False)
        auth.repository.set_active(user['id'], True)
        print(f"Password of '{user['username']}' changed; the account is active and unlocked.")
    elif args.command == "unlock":
        user = _find(auth, args.username)
        auth.repository.unlock(user['id'])
        print(f"'{user['username']}' unlocked.")
    elif args.command == "create-it":
        password = _ask_password(args.username)
        try:
            auth._create(args.username, args.full_name, Role.IT, password, must_change_password=False,
                         created_by="manage_accounts.py")
        except AuthError as e:
            raise SystemExit(str(e))
        print(f"IT account '{args.username}' created.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
