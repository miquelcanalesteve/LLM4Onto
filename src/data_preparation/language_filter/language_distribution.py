import ast
from collections import defaultdict
import pandas as pd

data = pd.read_excel("ontology_metrics_lang_2.xlsx")


# Diccionario para almacenar la suma total de valores por idioma
language_sums = defaultdict(float)
total_sum = 0  # Suma total de todos los valores de idiomas

# Recorrer cada registro en el DataFrame
for k in range(len(data)):
    data.at[k, "Languages"] = ast.literal_eval(data.at[k, "Languages"])
    
    # Filtrar los idiomas con valor < 0.1 o con clave vacía ('')
    keys_to_delete = [lang for lang in data.at[k, "Languages"] if lang == '']
    
    for key in keys_to_delete:
        del data.at[k, "Languages"][key]

    # Sumar los valores de los idiomas restantes
    for lang, value in data.at[k, "Languages"].items():
        language_sums[lang] += value
        total_sum += value

# Calcular la frecuencia de cada idioma
language_frequencies = {lang: (value / total_sum) * 100 for lang, value in language_sums.items()}

# Mostrar resultados ordenados por frecuencia
language_frequencies = dict(sorted(language_frequencies.items(), key=lambda x: x[1], reverse=True))

print("Frecuencia de cada idioma en porcentaje:")
for lang, freq in language_frequencies.items():
    print(f"{lang}: {freq:.2f}%")
