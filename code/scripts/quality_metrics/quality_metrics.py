from rdflib import Graph, RDF, RDFS, OWL, URIRef, Namespace, BNode
import pandas as pd
import os
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        use_auth_token="your_token"
    )


def count_tokens(text):
    """Tokenizes the text and returns the number of tokens."""
    tokens = tokenizer.tokenize(text)
    return len(tokens)


def identify_classes(g):
    """
    Identifies all resources that can be considered classes.
    Prioritizes explicit classes and only infers classes if no explicit ones are found.
    Removes datatypes defined as "rdfs:Datatype" or originating from "xsd:".

    Excludes certain RDF, RDFS, and OWL entities from being identified as classes.

    :param g: RDF graph.
    :return: Dictionary with classes as keys and identification details as values.
    """
    classes = {}
    XSD = Namespace("http://www.w3.org/2001/XMLSchema#")
    RDF_CLASS = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Class")

    # OWL terms to exclude
    OWL_EXCLUDES = {
        OWL.Ontology,
        OWL.Restriction,
        OWL.DeprecatedClass,
        OWL.ObjectProperty,
        OWL.TransitiveProperty,
        OWL.DatatypeProperty,
        OWL.FunctionalProperty,
        OWL.DeprecatedProperty,
        OWL.Thing,
        OWL.Nothing,
        OWL.AnnotationProperty,
        OWL.SymmetricProperty,
        OWL.InverseFunctionalProperty,
    }

    # Additional exclusions
    ADDITIONAL_EXCLUDES = {
        URIRef("http://www.w3.org/2000/01/rdf-schema#Class"),
        URIRef("http://www.w3.org/2002/07/owl#Class"),
        URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
    }

    # 1. Detect explicit classes via RDF.type OWL.Class
    for s, p, o in g.triples((None, RDF.type, OWL.Class)):
        classes[s] = {"type": "explicit (RDF.type OWL.Class)"}

    # 2. Detect explicit classes via RDF.type RDFS.Class
    for s, p, o in g.triples((None, RDF.type, RDFS.Class)):
        classes[s] = {"type": "explicit (RDF.type RDFS.Class)"}

    # 3. Detect explicit classes via RDF.type rdf:Class
    for s, p, o in g.triples((None, RDF.type, RDF_CLASS)):
        classes[s] = {"type": "explicit (RDF.type rdf:Class)"}

    # 4. Infer classes from Turtle syntax (resources after 'a')
    for s, p, o in g.triples((None, RDF.type, None)):
        if o not in classes:  # Only add if not explicitly defined as a class
            if (o in OWL_EXCLUDES or 
                o in ADDITIONAL_EXCLUDES or
                str(o).startswith(str(XSD)) or 
                o in {RDFS.Datatype, RDFS.Resource, RDFS.Literal}):
                continue  # Skip OWL exclusions, XSD datatypes, and RDFS exclusions
            classes[o] = {"type": "inferred (Turtle 'a')"}
    
    # Remove classes that are of type BNode
    for cls in list(classes.keys()):
        if isinstance(cls, BNode):
            del classes[cls]
    
    return classes

def calculate_totals_and_densities(concept_properties, g, n_classes):
    """
    Calculates the total, unique number of properties, densities, and non-taxonomic relationships.

    :param concept_properties: Dictionary with properties associated with each concept.
    :param g: RDF graph.
    :param n_classes: Total number of classes.
    :return: Dictionary with calculated metrics.
    """
    all_object_properties = set()
    all_data_annotation_properties = set()

    total_non_taxonomic_relations = 0
    non_taxonomic_relations_set = set()  # For the unique list

    for properties in concept_properties.values():
        for prop in properties["object_properties"]:
            all_object_properties.add(prop)
            if prop != RDFS.subClassOf:
                total_non_taxonomic_relations += 1  # Increment for each occurrence
                non_taxonomic_relations_set.add(prop)  # Add to the unique list

        all_data_annotation_properties.update(properties["data_annotation_properties"])

    total_object_properties = sum(len(properties["object_properties"]) for properties in concept_properties.values())
    total_data_annotation_properties = sum(len(properties["data_annotation_properties"]) for properties in concept_properties.values())

    # Densities
    property_density = (total_object_properties + total_data_annotation_properties) / n_classes if n_classes > 0 else 0
    non_taxonomic_density = total_non_taxonomic_relations / n_classes if n_classes > 0 else 0

    results = {
        "property_density": property_density,
        "non_taxonomic_density": non_taxonomic_density,  # Average non-taxonomic relationships per class
    }
    return results

def list_properties_by_concept(file_path):
    """
    Lists the properties (Object Properties and Data/Annotation Properties) associated with each concept in a Turtle file.

    :param file_path: Path to the TTL file.
    :return: Dictionary with concepts as keys and their categorized properties as values.
    """
    # Load the TTL file into a graph
    g = Graph()
    g.parse(file_path, format="turtle")

    # Identify classes in the graph
    classes = identify_classes(g)

    # Properties that should be explicitly excluded
    excluded_object_properties = {
        URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf"),
        RDF.type
    }

    # Dictionary to store properties by concept
    concept_properties = {}

    # Iterate over triples and associate properties with concepts
    for s, p, o in g.triples((None, None, None)):
        if s in classes:
            if s not in concept_properties:
                concept_properties[s] = {
                    "object_properties": [],
                    "data_annotation_properties": []
                }

            # Identify the property type based on the nature of the object (o)
            if isinstance(o, URIRef) and p not in excluded_object_properties:
                # If the object is a resource (IRI) and is not in the excluded properties, it is an Object Property
                concept_properties[s]["object_properties"].append(p)
            elif not isinstance(o, URIRef) and p not in excluded_object_properties:
                # If the object is a literal, it is a Data/Annotation Property
                concept_properties[s]["data_annotation_properties"].append(p)

    # Convert property sets to lists for easier readability
    return {
        concept: {
            "type": classes[concept]["type"],  # Add the class type to the result
            "object_properties": list(properties["object_properties"]),
            "data_annotation_properties": list(properties["data_annotation_properties"]),
        }
        for concept, properties in concept_properties.items()
    }, classes

def count_subclasses_and_average(g, total_classes):
    """
    Cuenta el número total de subclases y calcula el promedio de subclases por clase.

    :param g: Grafo RDF.
    :param total_classes: Número total de clases identificadas.
    :return: Diccionario con el total de subclases y el promedio por clase.
    """
    subclasses = {str(s) for s, p, o in g.triples((None, RDFS.subClassOf, None))}  # Use a set to ensure uniqueness
    total_subclasses = len(subclasses)
    average_subclasses = total_subclasses / total_classes if total_classes > 0 else 0

    return {"total_subclasses": total_subclasses, "average_subclasses_per_class": average_subclasses}

def count_tokens_in_file(file_path):
    """Reads a TTL file and counts the total tokens."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return count_tokens(content)

def process_ttl_file(file_path):
    """
    Process a single TTL file and calculate the metrics.

    :param file_path: Path to the TTL file.
    :return: Dictionary with calculated metrics for the file.
    """
    try:
        # Load the graph and calculate metrics
        g = Graph()
        g.parse(file_path, format="turtle")

        # Count total tokens
        total_tokens = count_tokens_in_file(file_path)

        # Total number of triples in the file
        total_triples = len(g)

        # Identify classes and properties
        concept_properties, classes = list_properties_by_concept(file_path)
        n_classes = len(classes)
        totals_and_densities = calculate_totals_and_densities(concept_properties, g, n_classes)
        subclass_metrics = count_subclasses_and_average(g, n_classes)

        # Return all metrics as a dictionary
        return {
            "File Name": os.path.basename(file_path),
            "Total Tokens": total_tokens,
            "Total Triples": total_triples,
            "Property Density": totals_and_densities['property_density'],
            "Average Non-Taxonomic Relations per Class": totals_and_densities['non_taxonomic_density'],
            "Average Subclasses per Class": subclass_metrics['average_subclasses_per_class'],
        }

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def min_max_normalize(series):
    """Applies Min-Max normalization to a Pandas Series."""
    return (series - series.min()) / (series.max() - series.min())

def add_normalized_columns(df):
    """Adds normalized columns for Property Density, Average Non-Taxonomic Relations, and Average Subclasses."""
    df["norm(Property Density by Class)"] = min_max_normalize(df["Property Density"])
    df["norm(Average Non-Taxonomic Relations by Class)"] = min_max_normalize(df["Average Non-Taxonomic Relations per Class"])
    df["norm(Average Subclasses by Class)"] = min_max_normalize(df["Average Subclasses per Class"])
    return df

def compute_quality_score(df):
    """Computes the Quality Score as the sum of normalized metrics."""
    df["Quality Score"] = (
        df["norm(Property Density by Class)"] +
        df["norm(Average Non-Taxonomic Relations by Class)"] +
        df["norm(Average Subclasses by Class)"]
    )
    return df

def compute_token_accumulation(df):
    """Computes cumulative token count and percentage accumulation."""
    df = df.sort_values(by="Quality Score", ascending=False)
    df["Token Count Acumulation"] = df["Total Tokens"].cumsum()
    df["Percentage of Token Count Acumulation"] = df["Token Count Acumulation"] / df["Token Count Acumulation"].iloc[-1] * 100
    return df

def process_ontology_metrics(file_path):
    """Loads, processes, and saves the ontology metrics dataframe with all computed metrics."""
    df = pd.read_excel(file_path)
    df = add_normalized_columns(df)
    df = compute_quality_score(df)
    df = compute_token_accumulation(df)
    df.to_excel(file_path, index=False)
    return df


if __name__ == "__main__":
    # Folder containing the TTL files
    # folder_path = "./../../data/ontology_repository" 
    folder_path = "./../../data/sample"  # Change this to the actual folder path
    output_excel = "./../../outputs/ontology_metrics.xlsx"

    # Initialize an empty DataFrame
    columns = [
            "File Name",
            "Total Tokens",
            "Total Triples",
            "Property Density",
            "Average Non-Taxonomic Relations per Class",
            "Average Subclasses per Class"]
    df = pd.DataFrame(columns=columns)

    # Process each TTL file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".ttl"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")
            
            # Process the file and calculate metrics
            metrics = process_ttl_file(file_path)
            
            if metrics:
                # Append the results to the DataFrame
                df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
                
                # Save the updated DataFrame to Excel
                df.to_excel(output_excel, index=False)
                # print(f"Metrics for {file_name} saved to {output_excel}")
    df = process_ontology_metrics(output_excel)

    print(f"Processing complete. Final results saved to {output_excel}")