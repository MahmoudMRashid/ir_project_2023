with open('Files\collection.txt', 'r',encoding="utf8") as file:
    data = file.read()

# Remove all non-ASCII characters
clean_data = ''.join(char for char in data if ord(char) < 128)

# Overwrite original file with clean data
with open('Files\collectionedit.txt', 'w') as file:
    file.write(clean_data)