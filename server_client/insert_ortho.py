import psycopg2

def insert_orthophoto(dbname, user, password, host, port, orthophoto_path, image_group_name):
    # Read the binary data from the image file
    with open(orthophoto_path, 'rb') as file:
        orthophoto_data = file.read()

    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    cursor = conn.cursor()

    # Insert the orthophoto into the database
    insert_query = """
    INSERT INTO orthophotos (orthophoto, image_group_name)
    VALUES (%s, %s)
    """
    cursor.execute(insert_query, (psycopg2.Binary(orthophoto_data), image_group_name))

    # Commit the transaction
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()

# Parameters
dbname = 'mapdb'
user = 'postgres'
password = '123'
host = 'localhost'
port = '5432'
orthophoto_path = '/home/user/Desktop/Maps_DONE/Brighton_beach_images/result/01.tif'
image_group_name = 'second'

# Insert the orthophoto
insert_orthophoto(dbname, user, password, host, port, orthophoto_path, image_group_name)

