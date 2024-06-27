import sqlite3

def remove_all_data(db_file, table_name):
  """
  This function removes all data from a table in a SQLite database.

  Args:
    db_file: The path to the SQLite database file.
    table_name: The name of the table to delete data from.
  """
  conn = sqlite3.connect(db_file)
  cursor = conn.cursor()

  try:
    cursor.execute(f"DELETE FROM {table_name}")
    conn.commit()
    print(f"All data from table '{table_name}' deleted successfully.")
  except sqlite3.Error as error:
    print(f"An error occurred while deleting data: {error}")
  finally:
    cursor.close()
    conn.close()

if __name__ == "__main__":
  # Get user input for the table name
  table_name = input("Enter the name of the table to delete data from: ")

  # Call the function to delete data
  remove_all_data("customer_faces_data.db", table_name)
