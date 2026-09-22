"""Local MySQL store for title/content rows (no files)."""

import argparse
import hashlib
import os

import pymysql
from pymysql.cursors import DictCursor

DEFAULTS = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "docsearch",
    "password": "docsearch",
    "database": "docsearch",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  title VARCHAR(512) NOT NULL,
  content TEXT NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

SEED_ROWS = [
    {
        "title": "Helium Bakery Robots Open in Reykjavik",
        "content": (
            "A downtown bakery installed helium-cooled pastry robots that keep "
            "laminated dough at a constant minus two degrees while folding it. "
            "Night bakers say the machines hum in a narrow band that also keeps "
            "the proofing cabinets frost-free."
        ),
    },
    {
        "title": "City Bicycle Ledger Still Uses Paper Stamps",
        "content": (
            "The municipal bicycle-share office records every checkout with a "
            "dated paper stamp in a bound ledger. Staff refused a cloud app "
            "because winter outages left riders unable to return bikes last year."
        ),
    },
    {
        "title": "Subsea Kelp Fiber Carries Winter Power",
        "content": (
            "Engineers floated a kelp-derived fiber cable between two fjord "
            "substations. The wet fiber stays flexible in ice and is meant to "
            "carry village loads when the overland line ices over."
        ),
    },
    {
        "title": "Otters Inspect Sluice Gates on Night Shift",
        "content": (
            "A river authority trained a small team of otters to tug marker "
            "rings on sluice hinges after dark. Handlers log each inspection "
            "in a waterproof notebook rather than a phone, which fogs in the spray."
        ),
    },
    {
        "title": "Ceramic Postage Melts in Rain as Anti-Fraud",
        "content": (
            "The island post office fired thin ceramic stamps that dissolve "
            "into chalk when soaked. A letter that arrives with an intact stamp "
            "could not have been steamed open in transit."
        ),
    },
    {
        "title": "Thunderstorm Archive Prices Crop Insurance",
        "content": (
            "An inland co-op stores labeled recordings of local thunderstorms "
            "and plays them back to score hail risk. Underwriters listen for "
            "the same crackle pattern before they set the next season's premium."
        ),
    },
    {
        "title": "Glass Harmonica Keeps the Factory Clock",
        "content": (
            "A glassworks replaced its electric shift bell with a glass "
            "harmonica that sounds every quarter hour. The wet-finger pitch "
            "stays stable when the furnace room loses mains power."
        ),
    },
    {
        "title": "Desert Tram Harvests Static for Lamps",
        "content": (
            "A short tram line in the salt flats trails a comb that strips "
            "static from the rails and feeds streetlamps at each stop. Riders "
            "notice a faint ozone smell only on the driest afternoons."
        ),
    },
    {
        "title": "Fermented Ink Flags Expired Contracts",
        "content": (
            "A notary uses fermented oak-gall ink that shifts from brown to "
            "green when a stored contract passes its written expiry date. "
            "Clerks check the color before they accept a filing."
        ),
    },
    {
        "title": "Balloon Post Links Mountain Clinics",
        "content": (
            "Two high clinics exchange lab slips by tethered balloon when the "
            "switchback road is closed. Each capsule carries a paper card and "
            "a dried-ink blot used to confirm the sender."
        ),
    },
    {
        "title": "Salt-Block Servers Run Village Weather",
        "content": (
            "A coastal village stacked salt-block computers in a shed to run "
            "a three-kilometer weather model. The blocks wick heat and need "
            "only a hand-crank fan during the afternoon glare."
        ),
    },
    {
        "title": "Tuning-Fork Choir Encodes Tide Tables",
        "content": (
            "Harbor pilots keep a rack of stamped tuning forks whose beat "
            "notes encode the week's tide heights. A newcomer learns the "
            "intervals instead of unfolding a wet paper chart on deck."
        ),
    },
]


def settings(admin=False):
    prefix = "MYSQL_ADMIN_" if admin else "MYSQL_"
    host = os.environ.get(f"{prefix}HOST") or os.environ.get("MYSQL_HOST") or DEFAULTS["host"]
    port = int(os.environ.get(f"{prefix}PORT") or os.environ.get("MYSQL_PORT") or DEFAULTS["port"])
    if admin:
        user = os.environ.get("MYSQL_ADMIN_USER", "root")
        password = os.environ.get("MYSQL_ADMIN_PASSWORD", "")
        database = os.environ.get("MYSQL_ADMIN_DATABASE") or None
    else:
        user = os.environ.get("MYSQL_USER", DEFAULTS["user"])
        password = os.environ.get("MYSQL_PASSWORD", DEFAULTS["password"])
        database = os.environ.get("MYSQL_DATABASE", DEFAULTS["database"])
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
        "autocommit": False,
    }


def connect(admin=False):
    cfg = {key: value for key, value in settings(admin=admin).items() if value is not None}
    return pymysql.connect(**cfg)


def ensure_environment():
    """Create the database, app user, and table on the local MySQL server."""
    app = settings()
    admin = connect(admin=True)
    try:
        with admin.cursor() as cur:
            cur.execute(
                "CREATE DATABASE IF NOT EXISTS `{}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci".format(app["database"])
            )
            user = app["user"]
            if not user.isidentifier():
                raise ValueError(f"Unsafe MySQL user name: {user}")
            for host in ("localhost", "127.0.0.1"):
                cur.execute(
                    f"CREATE USER IF NOT EXISTS '{user}'@'{host}' IDENTIFIED BY %s",
                    (app["password"],),
                )
                cur.execute(
                    f"GRANT ALL PRIVILEGES ON `{app['database']}`.* TO '{user}'@'{host}'"
                )
            cur.execute("FLUSH PRIVILEGES")
        admin.commit()
    finally:
        admin.close()
    conn = connect()
    try:
        init_schema(conn)
    finally:
        conn.close()


def init_schema(conn):
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()


def row_digest(title, content):
    payload = f"{title}\n\n{content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def embed_text(title, content):
    return f"{(title or '').strip()}\n\n{(content or '').strip()}".strip()


def fetch_records(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, content FROM records ORDER BY id")
        return list(cur.fetchall())


def table_snapshot(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n, "
            "COALESCE(UNIX_TIMESTAMP(MAX(updated_at)), 0) AS ts, "
            "COALESCE(MAX(id), 0) AS max_id "
            "FROM records"
        )
        row = cur.fetchone()
    return (int(row["n"]), int(row["ts"]), int(row["max_id"]))


def insert_record(conn, title, content):
    title = (title or "").strip()
    content = (content or "").strip()
    if not title:
        raise ValueError("Title is empty.")
    if not content:
        raise ValueError("Content is empty.")
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO records (title, content) VALUES (%s, %s)",
            (title, content),
        )
        record_id = cur.lastrowid
    conn.commit()
    return record_id


def delete_record(conn, record_id):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM records WHERE id = %s", (int(record_id),))
    conn.commit()


def seed_if_empty(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM records")
        count = cur.fetchone()["n"]
        if count:
            return 0
        cur.executemany(
            "INSERT INTO records (title, content) VALUES (%s, %s)",
            [(row["title"], row["content"]) for row in SEED_ROWS],
        )
    conn.commit()
    return len(SEED_ROWS)


def init_db(seed=True):
    ensure_environment()
    conn = connect()
    try:
        init_schema(conn)
        inserted = seed_if_empty(conn) if seed else 0
        rows = fetch_records(conn)
    finally:
        conn.close()
    return inserted, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true", help="Create the database, table, and seed rows")
    parser.add_argument("--list", action="store_true", help="Print id and title for every row")
    parser.add_argument("--add", action="store_true", help="Insert one row")
    parser.add_argument("--title", default="")
    parser.add_argument("--content", default="")
    parser.add_argument("--delete", type=int, default=None, help="Delete a row by id")
    args = parser.parse_args()
    if not (args.init or args.list or args.add or args.delete is not None):
        parser.error("Use --init, --list, --add, or --delete")
    try:
        if args.init:
            inserted, rows = init_db(seed=True)
            print(f"MySQL {settings()['database']}: {len(rows)} rows ({inserted} seeded).")
        conn = connect()
        try:
            init_schema(conn)
            if args.add:
                record_id = insert_record(conn, args.title, args.content)
                print(f"Inserted id={record_id}")
            if args.delete is not None:
                delete_record(conn, args.delete)
                print(f"Deleted id={args.delete}")
            if args.list:
                for row in fetch_records(conn):
                    print(f"{row['id']}\t{row['title']}")
        finally:
            conn.close()
    except (ValueError, OSError, pymysql.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
