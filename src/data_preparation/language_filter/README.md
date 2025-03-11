


Identify language puede tardar muchas hora en ejecutarse, por lo que tiene sentido ejecutarlo independientemente.

Con la salida de select_ontologies_by_language.py es necesario una revisión manual porque hay ontologias que directamente no tienen literales ni en inglés ni en otro idioma, son solo URIs y esas las queremos en el entrenamiento. 

En el dataset de dbpedia archivo descargado en x fecha se eliminaron las ontologías:
- bgt--def_type=parsed.ttl
- def--nen3610_type=parsed.ttl 
- datosabiertos--kos--sector-publico--empleo--grupo-profesional_type=parsed.ttl 


Si se eliminan manualmente del fichero "outputs.xlsx" o se crea una copia de ese archivo y se eliminan las ontologias correspondientes para explorar el resto del corpus con language_distribution.py