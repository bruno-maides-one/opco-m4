## Exemple de prompt

---

créer une fonction python pour trier un tableau

```python
def sort_array(arr):
  """
  Trie un tableau (liste) en ordre croissant.

  Args:
    arr: La liste à trier.

  Returns:
    Une nouvelle liste triée.
  """
  return sorted(arr)

# Exemple d'utilisation:
my_list = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_list = sort_array(my_list)
print(f"Liste originale: {my_list}")
print(f"Liste triée: {sorted_list}")

# Pour trier la liste en place (sans créer une nouvelle liste):
my_list.sort()
print(f"Liste triée en place: {my_list}")
```

---

compter le nombre d'occurence unique d'un élément dans un tableau

```python
arr = [1,3,2,7,4,6,2,8,1]

result = {}

for item in arr:
    sitem = str(item)
    if str(item) in result:
        result[sitem] = result[sitem] + 1
    else:
        result[sitem] = 1

print(result)
```




