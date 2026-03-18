import sqlite3 as sql
from datetime import datetime


def getLogs():
    con = sql.connect("databaseFiles/database.db")
    cur = con.cursor()
    cur.execute("SELECT rowid, * FROM logs ORDER BY datetime(entry_time) DESC")
    rows = cur.fetchall()
    con.close()

    # format dates for display (ai generated)
    formatted_rows = []
    for row in rows:
        formatted_row = list(row)
        # Format start_time (index 3), end_time (index 4), entry_time (index 5)
        for idx in [3, 4, 5]:
            if formatted_row[idx]:
                try:
                    dt = datetime.fromisoformat(formatted_row[idx])
                    # Format as "16 Dec 2025, 2:39 PM"
                    formatted_row[idx] = dt.strftime("%d %b %Y, %I:%M %p")
                except:
                    pass  # Keep original if parsing fails
        formatted_rows.append(tuple(formatted_row))

    return formatted_rows


def insertLog(
    developer, project, start_time, end_time, entry_time, time_worked, repo, notes
):
    con = sql.connect("databaseFiles/database.db")
    cur = con.cursor()
    try:
        cur.execute(
            "INSERT INTO logs (developer,project,start_time,end_time,entry_time,time_worked,repo,notes) VALUES (?,?,?,?,?,?,?,?)",
            (
                developer,
                project,
                start_time,
                end_time,
                entry_time,
                time_worked,
                repo,
                notes,
            ),
        )
        con.commit()
        con.close()
        return (True, "Log added")
    except sql.IntegrityError:
        con.close()
        return (False, "Error, log not added")
