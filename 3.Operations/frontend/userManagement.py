import sqlite3 as sql
import bcrypt


### example
def getUsers():
    con = sql.connect("databaseFiles/database.db")
    cur = con.cursor()
    cur.execute("SELECT * FROM id7-tusers")
    con.close()
    return cur


def insertUser(email, password):
    con = sql.connect("databaseFiles/database.db")
    cur = con.cursor()
    try:
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        cur.execute(
            "INSERT INTO users (email,password) VALUES (?,?)",
            (email, hashed.decode("utf-8")),
        )
        con.commit()
        con.close()
        return (True, "Inserted")
    except sql.IntegrityError:
        con.close()
        return (False, "Email already exists")


def verifyUser(email, password):
    con = sql.connect("databaseFiles/database.db")
    cur = con.cursor()
    cur.execute("SELECT email,password FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    con.close()
    if row == None:
        return (False, "Email isn't registered")
    else:
        stored_password = row[1]
        if bcrypt.checkpw(password.encode("utf-8"), stored_password.encode("utf-8")):
            return (True, "Signed in")
        else:
            return (False, "Wrong password")
