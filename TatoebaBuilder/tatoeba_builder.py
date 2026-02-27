import hashlib
import re
import sqlite3
import csv
import unicodedata
from TatoebaBuilder.Protobuf import links_data_pb2


def create_and_define_database():
    conn = sqlite3.connect("TatoebaBuilder/tatoeba.db")

    cursor = conn.cursor()

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS sentences(
            id INTEGER PRIMARY KEY,
            lang CHAR(3),
            text VARCHAR
        );"""
    )

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS links(
            srcLang CHAR(3),
            tgtLang CHAR(3),
            data MEDIUMBLOB,
            PRIMARY KEY (srcLang, tgtLang)
        );"""
    )

    return conn, cursor


def fill_sentences(cursor: sqlite3.Cursor):
    #we fill the sentences table with the info in sentences.csv
    with open('TatoebaBuilder/sentences.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='	', quotechar='|')
        for row in spamreader:
            #print("id: "+row[0]+", lang: "+row[1]+", text: "+row[2])
            cursor.execute(
                "INSERT INTO sentences VALUES (?, ?, ?)",
                (int(row[0]), row[1], row[2])
            )




def fill_links(cursor: sqlite3.Cursor):
    """
    we create a dictionary with this structure:

    {
        srcLang-tgtLang : {
            hashSentence : [(linkSrc, linkTgt), ...],
            ...
        },
        ...
    }

    But using the types of links_data_pb2, to make the conversion of the value of each lang pair to proto easier:

    {
        srcLang-tgtLang : DataMap{
            hashSentence : PairList[Pair(linkSrc, linkTgt), ...],
            ...
        },
        ...
    }
    """

    data: dict[str, links_data_pb2.DataMap] = {}

    # we iterate each row of links.cvs
    with open('TatoebaBuilder/links.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='	', quotechar='|')
        skippedCount = 0
        for row in spamreader:
            # for each row we fetch the data of srcSentence and tgtSentence
            res = cursor.execute("SELECT * FROM sentences WHERE id IN ("+row[0]+", "+row[1]+");")
            sentences = res.fetchall()
            if(len(sentences) < 2):
                print("Skipped link ("+row[0]+", "+row[1]+")")
                skippedCount = skippedCount + 1
                continue
            srcSentenceId = sentences[0][0]
            srcLang = sentences[0][1]
            srcSentenceText = sentences[0][2]

            tgtSentenceId = sentences[1][0]
            tgtLang = sentences[1][1]
            tgtSentenceText: str = sentences[1][2]

            # then we insert the srcLang and tgtLang combo key in the dictionary and associate to it an empty dictionary (if it doesn't exists yet),
            if((srcLang+"-"+tgtLang) not in data):
                data[srcLang+"-"+tgtLang] = links_data_pb2.DataMap()
            # then, we calculate the hash of the srcSentence and insert, in the dictionary associated, the hash as a new key and associate to it a new array containing the couple (srcLink, tgtLink) (if the hash key already exists we add this couple to the array)
            srcSentenceTextNormalized = _normalize_text(srcSentenceText)
            hash = hashlib.shake_256(srcSentenceTextNormalized.encode("utf-8")).hexdigest(8)
            data[srcLang+"-"+tgtLang].data[hash].items.add(srcSentence=srcSentenceId, tgtSentence=tgtSentenceId)

    print("Total skiped links: "+str(skippedCount))
    # now we insert the data contained in the dictionary in the table links of the db
    for key, value in data.items():
        # to do this we simply create a new row for each language combo of the dict, and insert in it the srcLang, tgtLang and the proto binary of the value associated to the lang combo.
        srcLang, tgtLang = key.split("-")
        cursor.execute(
            "INSERT INTO links VALUES (?, ?, ?)",
            (srcLang, tgtLang, value.SerializeToString())
        )



def remove_lang_from_sentences_db(cursor: sqlite3.Cursor):
    # we remove the lang column from the sentences table of the db
    cursor.execute("ALTER TABLE sentences DROP COLUMN lang;")
    cursor.execute("VACUUM")


def _normalize_text(text: str) -> str:
    if text is None:
        return None

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Standardize punctuation
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Trim
    text = text.strip()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Case normalization
    text = text.casefold()

    return text


if __name__ == '__main__':
    connection, cursor = create_and_define_database()
    fill_sentences(cursor)
    connection.commit()
    fill_links(cursor)
    connection.commit()
    remove_lang_from_sentences_db(cursor)
    connection.commit()
    connection.close()
    
    #hash = hashlib.shake_256(_normalize_text("Ciao, come stai?").encode("utf-8")).hexdigest(8)
    #print(hash)