def reading_file(path_file):
    """
    Reads a csv file, if a path given. Example:
    >>> type(reading_file("iris.csv"))
    list
    """
    dataset = []
    with open(path_file, "r", encoding="utf-8") as file:
        file.readline()
        for line in file:
            line = line.rstrip()
            line = line.replace("Setosa", "0")
            line= line.replace("Versicolor", "1")
            line = line.replace("Virginica", "2")
            line = line.split(",")
            line[-1] = int(line[-1][1])
            line = list(map(lambda x: float(x), line))
            line[-1] = int(line[-1])
            dataset.append(line)
    return dataset


