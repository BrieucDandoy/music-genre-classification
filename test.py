liste = list(range(5))
liste = [5*[item] for item in liste]
liste2 = [item for subliste in liste for item in subliste]
print(liste2)