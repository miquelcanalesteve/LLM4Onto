from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_and_generate(local_model_path, prompt, gpu_id=4, max_length=450):

    # Verificar si la GPU específica está disponible
    if torch.cuda.device_count() > gpu_id and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Usando GPU {gpu_id} para la generación de texto.")
    else:
        device = torch.device("cpu")
        print(f"GPU {gpu_id} no está disponible. Usando CPU.")

    # Cargar el tokenizador y el modelo desde la ruta local
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(local_model_path).to(device)

    # Tokenizar el prompt y moverlo al dispositivo
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generar texto a partir del prompt
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
        #outputs = model.generate(**inputs, max_length=max_length, do_sample=False)


    # Decodificar la salida generada
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

prompts_list=["""###  http://edamontology.org/citation
:citation rdf:type owl:AnnotationProperty ;
          :created_in "1.13" ;
          oboLegacy:is_metadata_tag "true"^^xsd:boolean ;
          oboInOwl:hasBroadSynonym "Publication reference" ;
          oboInOwl:hasDefinition "'Citation' concept property ('citation' metadata tag) contains a dereferenceable URI, preferably including a DOI, pointing to a citeable publication of the given data format." ;
          oboInOwl:hasRelatedSynonym "Publication" ;
          oboInOwl:inSubset :properties ;
          rdfs:label "Citation" .


###  http:/""",
          """###  http://edamontology.org/citation
:citation rdf:type owl:AnnotationProperty ;
          :created_in "1.13" ;
          oboLegacy:is_metadata_tag "true"^^xsd:boolean ;
          oboInOwl:hasBroadSynonym "Publication reference" ;
          oboInOwl:hasDefinition "'Citation' concept property ('citation' metadata tag) contains a dereferenceable URI, preferably including a DOI, pointing to a citeable publication of the given data format." ;
          oboInOwl:hasRelatedSynonym "Publication" ;
          oboInOwl:inSubset :properties ;
          rdfs:label "Citation" .


###  http:/""",
          """###  http://edamontology.org/citation
:citation rdf:type owl:AnnotationProperty ;
          :created_in "1.13" ;
          oboLegacy:is_metadata_tag "true"^^xsd:boolean ;
          oboInOwl:hasBroadSynonym "Publication reference" ;
          oboInOwl:hasDefinition "'Citation' concept property ('citation' metadata tag) contains a dereferenceable URI, preferably including a DOI, pointing to a citeable publication of the given data format." ;
          oboInOwl:hasRelatedSynonym "Publication" ;
          oboInOwl:inSubset :properties ;
          rdfs:label "Citation" .


###  http:/""",
          """:has_identifier rdf:type owl:ObjectProperty ;
                owl:inverseOf :is_identifier_of ;
                rdfs:domain :data_0006 ;
                rdfs:range :data_0842 ;
                oboLegacy:is_anti_symmetric "false" ;
                oboLegacy:is_reflexive "false" ;
                oboLegacy:is_symmetric "false" ;
                oboLegacy:transitive_over "OBO_REL:is_a" ;
                oboInOwl:hasDefinition "'A has_identifier B' defines for the subject A, that it has the object B as its identifier." ;
                oboInOwl:inSubset :properties ;
                oboInOwl:isC""",
            """:has_identifier rdf:type owl:ObjectProperty ;
                owl:inverseOf :is_identifier_of ;
                rdfs:domain :data_0006 ;
                rdfs:range :data_0842 ;
                oboLegacy:is_anti_symmetric "false" ;
                oboLegacy:is_reflexive "false" ;
                oboLegacy:is_symmetric "false" ;
                oboLegacy:transitive_over "OBO_REL:is_a" ;
                oboInOwl:hasDefinition "'A has_identifier B' defines for the subject A, that it has the object B as its identifier." ;
                oboInOwl:inSubset :properties ;
                oboInOwl:isC""",
            """:has_identifier rdf:type owl:ObjectProperty ;
                owl:inverseOf :is_identifier_of ;
                rdfs:domain :data_0006 ;
                rdfs:range :data_0842 ;
                oboLegacy:is_anti_symmetric "false" ;
                oboLegacy:is_reflexive "false" ;
                oboLegacy:is_symmetric "false" ;
                oboLegacy:transitive_over "OBO_REL:is_a" ;
                oboInOwl:hasDefinition "'A has_identifier B' defines for the subject A, that it has the object B as its identifier." ;
                oboInOwl:inSubset :properties ;
                oboInOwl:isC""",
            """:data_0006 rdf:type owl:Class ;
           owl:disjointWith :format_1915 ,
                            :operation_0004 ,
                            :topic_0003 ,
                            owl:DeprecatedClass ;
           :created_in "beta12orEarlier" ;
           :notRecommendedForAnnotation "true"^^xsd:boolean ;
           oboInOwl:hasDefinition "Information, represented in an information artefact (data record) that is 'understandable' by dedicated computational tools that can use the data as input or produce it as output." ;
           oboInOwl:hasExactSynonym "Data record" ;
           oboInOwl:hasNarrowSynonym "Data set" ,
                                     "Datum""",
           """:data_0006 rdf:type owl:Class ;
           owl:disjointWith :format_1915 ,
                            :operation_0004 ,
                            :topic_0003 ,
                            owl:DeprecatedClass ;
           :created_in "beta12orEarlier" ;
           :notRecommendedForAnnotation "true"^^xsd:boolean ;
           oboInOwl:hasDefinition "Information, represented in an information artefact (data record) that is 'understandable' by dedicated computational tools that can use the data as input or produce it as output." ;
           oboInOwl:hasExactSynonym "Data record" ;
           oboInOwl:hasNarrowSynonym "Data set" ,
                                     "Datum""",
           """:data_0006 rdf:type owl:Class ;
           owl:disjointWith :format_1915 ,
                            :operation_0004 ,
                            :topic_0003 ,
                            owl:DeprecatedClass ;
           :created_in "beta12orEarlier" ;
           :notRecommendedForAnnotation "true"^^xsd:boolean ;
           oboInOwl:hasDefinition "Information, represented in an information artefact (data record) that is 'understandable' by dedicated computational tools that can use the data as input or produce it as output." ;
           oboInOwl:hasExactSynonym "Data record" ;
           oboInOwl:hasNarrowSynonym "Data set" ,
                                     "Datum""",
           """pmd:hasIdentifier rdf:type owl:ObjectProperty ;
                  rdfs:domain mds:BeamlineConfiguration ,
                              mds:CroppingMethod ,
                              mds:DarkCorrectionMethod1 ,
                              mds:DarkCorrectionMethod2 ,
                              mds:DetectorCalibration ,
                              mds:DiffractionFrame ,
                              mds:GravityMethod ,
                              mds:Integration ,
                              mds:Modeling ,
                              mds:PhotovoltaicBacksheet ,
                              mds:PhotovoltaicCell ,
                              mds:PhotovoltaicInverter ,
                              mds:PhotovoltaicModule ,
                              mds:PhotovoltaicSite ,
                              mds:PixelwiseCorrelation ,
                              mds:RingSamplingMethod ,
                              m""",
            """pmd:hasIdentifier rdf:type owl:ObjectProperty ;
                  rdfs:domain mds:BeamlineConfiguration ,
                              mds:CroppingMethod ,
                              mds:DarkCorrectionMethod1 ,
                              mds:DarkCorrectionMethod2 ,
                              mds:DetectorCalibration ,
                              mds:DiffractionFrame ,
                              mds:GravityMethod ,
                              mds:Integration ,
                              mds:Modeling ,
                              mds:PhotovoltaicBacksheet ,
                              mds:PhotovoltaicCell ,
                              mds:PhotovoltaicInverter ,
                              mds:PhotovoltaicModule ,
                              mds:PhotovoltaicSite ,
                              mds:PixelwiseCorrelation ,
                              mds:RingSamplingMethod ,
                              m""",
            """pmd:hasIdentifier rdf:type owl:ObjectProperty ;
                  rdfs:domain mds:BeamlineConfiguration ,
                              mds:CroppingMethod ,
                              mds:DarkCorrectionMethod1 ,
                              mds:DarkCorrectionMethod2 ,
                              mds:DetectorCalibration ,
                              mds:DiffractionFrame ,
                              mds:GravityMethod ,
                              mds:Integration ,
                              mds:Modeling ,
                              mds:PhotovoltaicBacksheet ,
                              mds:PhotovoltaicCell ,
                              mds:PhotovoltaicInverter ,
                              mds:PhotovoltaicModule ,
                              mds:PhotovoltaicSite ,
                              mds:PixelwiseCorrelation ,
                              mds:RingSamplingMethod ,
                              m""",
            """###  https://cwrusdle.bitbucket.io/mds#Architecture
mds:Architecture rdf:type owl:Class ;
                 rdfs:subClassOf mds:ModelingParameters ;
                 rdfs:label "Architecture" ;
                 skos:altLabel "" ;
                 skos:definition "Architecture of the model." .


###  https://cwrusdle.bitbucket.io/mds#AzimuthRange
mds:AzimuthRange rdf:type owl:Class ;
                 rdfs:subClassOf mds:IntegrationOptions ;
                 rdfs:label "AzimuthRange" ;
                 skos:altLabel "" ;
                 skos:definition "The azimuthal integration range." .
                 
###  https://""",
            """###  https://cwrusdle.bitbucket.io/mds#Architecture
mds:Architecture rdf:type owl:Class ;
                 rdfs:subClassOf mds:ModelingParameters ;
                 rdfs:label "Architecture" ;
                 skos:altLabel "" ;
                 skos:definition "Architecture of the model." .


###  https://cwrusdle.bitbucket.io/mds#AzimuthRange
mds:AzimuthRange rdf:type owl:Class ;
                 rdfs:subClassOf mds:IntegrationOptions ;
                 rdfs:label "AzimuthRange" ;
                 skos:altLabel "" ;
                 skos:definition "The azimuthal integration range." .
                 
###  https://""",
            """###  https://cwrusdle.bitbucket.io/mds#Architecture
mds:Architecture rdf:type owl:Class ;
                 rdfs:subClassOf mds:ModelingParameters ;
                 rdfs:label "Architecture" ;
                 skos:altLabel "" ;
                 skos:definition "Architecture of the model." .


###  https://cwrusdle.bitbucket.io/mds#AzimuthRange
mds:AzimuthRange rdf:type owl:Class ;
                 rdfs:subClassOf mds:IntegrationOptions ;
                 rdfs:label "AzimuthRange" ;
                 skos:altLabel "" ;
                 skos:definition "The azimuthal integration range." .
                 
###  https://""",
            """pmd:composedOf rdf:type owl:ObjectProperty ;
               rdfs:domain mds:BeamlineConfiguration ,
                           mds:DetectorResolution ,
                           mds:DiffractionFrame ,
                           mds:PhotovoltaicModule ,
                           mds:XrayRecipe ,
                           pmd:Furnace ;
               rdfs:range mds:BeamlineConfiguration ,
                          mds:Calibration ,
                          mds:DataCollection ,
                          mds:DetectorParameters ,
                          mds:ExperimentType ,
                          mds:PhotovoltaicBacksheet ,
                          mds:PhotovoltaicCell ,
                          mds:PhotovoltaicInverter ,
                          mds:PixelHeight ,
                          mds:PixelWidth ,
                          mds:XrayRecipe""",
            """pmd:composedOf rdf:type owl:ObjectProperty ;
               rdfs:domain mds:BeamlineConfiguration ,
                           mds:DetectorResolution ,
                           mds:DiffractionFrame ,
                           mds:PhotovoltaicModule ,
                           mds:XrayRecipe ,
                           pmd:Furnace ;
               rdfs:range mds:BeamlineConfiguration ,
                          mds:Calibration ,
                          mds:DataCollection ,
                          mds:DetectorParameters ,
                          mds:ExperimentType ,
                          mds:PhotovoltaicBacksheet ,
                          mds:PhotovoltaicCell ,
                          mds:PhotovoltaicInverter ,
                          mds:PixelHeight ,
                          mds:PixelWidth ,
                          mds:XrayRecipe""",
            """pmd:composedOf rdf:type owl:ObjectProperty ;
               rdfs:domain mds:BeamlineConfiguration ,
                           mds:DetectorResolution ,
                           mds:DiffractionFrame ,
                           mds:PhotovoltaicModule ,
                           mds:XrayRecipe ,
                           pmd:Furnace ;
               rdfs:range mds:BeamlineConfiguration ,
                          mds:Calibration ,
                          mds:DataCollection ,
                          mds:DetectorParameters ,
                          mds:ExperimentType ,
                          mds:PhotovoltaicBacksheet ,
                          mds:PhotovoltaicCell ,
                          mds:PhotovoltaicInverter ,
                          mds:PixelHeight ,
                          mds:PixelWidth ,
                          mds:XrayRecipe""",
            """<http://sweetontology.net/matrRockIgneous/CinderCone> rdfs:comment "A volcanic cone built entirely of loose fragmented material (pyroclastics.)"@en .


<http://sweetontology.net/matrRockIgneous/CompositeCone> rdfs:comment "A steep volcanic cone built by both lava flows and pyroclastic eruptions"@en .


<http://sweetontology.net/matrRockIgneous/Dacite> rdfs:comment "Volcanic rock (or lava) that characteristically is light in color and contains 62% to 69% silica and moderate a mounts of sodium and potassium."@en .


<http://sweetontology.net/matr""",
            """<http://sweetontology.net/matrRockIgneous/CinderCone> rdfs:comment "A volcanic cone built entirely of loose fragmented material (pyroclastics.)"@en .


<http://sweetontology.net/matrRockIgneous/CompositeCone> rdfs:comment "A steep volcanic cone built by both lava flows and pyroclastic eruptions"@en .


<http://sweetontology.net/matrRockIgneous/Dacite> rdfs:comment "Volcanic rock (or lava) that characteristically is light in color and contains 62% to 69% silica and moderate a mounts of sodium and potassium."@en .


<http://sweetontology.net/matr""",
            """<http://sweetontology.net/matrRockIgneous/CinderCone> rdfs:comment "A volcanic cone built entirely of loose fragmented material (pyroclastics.)"@en .


<http://sweetontology.net/matrRockIgneous/CompositeCone> rdfs:comment "A steep volcanic cone built by both lava flows and pyroclastic eruptions"@en .


<http://sweetontology.net/matrRockIgneous/Dacite> rdfs:comment "Volcanic rock (or lava) that characteristically is light in color and contains 62% to 69% silica and moderate a mounts of sodium and potassium."@en .


<http://sweetontology.net/matr""",
            """<http://sweetontology.net/matrRockIgneous/Hypabyssal> rdfs:comment "Hypabyssal are igneous rocks formed at a depth in between the plutonic and volcanic rocks. They are characterized by their porphyritic nature (porphyry). They consist of phenocrysts embedded in a fine-grained groundmass. [Wikipedia]"@en .


<http://sweetontology.net/matrRockIgneous/IgneousRock> rdfs:comment "By definition, all igneous rock is formed from magma [Wikipedia]"@en .


<http://sweetontology.net/matrRockIgneous/IntrusiveRock> rdfs:comment "Beneath """,
            """<http://sweetontology.net/matrRockIgneous/Hypabyssal> rdfs:comment "Hypabyssal are igneous rocks formed at a depth in between the plutonic and volcanic rocks. They are characterized by their porphyritic nature (porphyry). They consist of phenocrysts embedded in a fine-grained groundmass. [Wikipedia]"@en .


<http://sweetontology.net/matrRockIgneous/IgneousRock> rdfs:comment "By definition, all igneous rock is formed from magma [Wikipedia]"@en .


<http://sweetontology.net/matrRockIgneous/IntrusiveRock> rdfs:comment "Beneath """,
            """<http://sweetontology.net/matrRockIgneous/Hypabyssal> rdfs:comment "Hypabyssal are igneous rocks formed at a depth in between the plutonic and volcanic rocks. They are characterized by their porphyritic nature (porphyry). They consist of phenocrysts embedded in a fine-grained groundmass. [Wikipedia]"@en .


<http://sweetontology.net/matrRockIgneous/IgneousRock> rdfs:comment "By definition, all igneous rock is formed from magma [Wikipedia]"@en .


<http://sweetontology.net/matrRockIgneous/IntrusiveRock> rdfs:comment "Beneath """,
            """<http://sweetontology.net/propDiffusivity/Confluence> rdfs:comment "The rate at which adjacent flow is converging along an axis oriented normal to the flow at the point in question."@en .


<http://sweetontology.net/propDiffusivity/Diffluence> rdfs:comment "The rate at which adjacent flow diverges along an axis oriented normal to the flow at the point in question; the opposite of confluence."@en .


<http://sweetontology.net/propDiffusivity/EddyDiffusivity> rdfs:comment "The exchange coefficient for the diffusion of a conservative property by eddies in a turbulent flow."@en .


<http://sweetontology.net""",
            """<http://sweetontology.net/propDiffusivity/Confluence> rdfs:comment "The rate at which adjacent flow is converging along an axis oriented normal to the flow at the point in question."@en .


<http://sweetontology.net/propDiffusivity/Diffluence> rdfs:comment "The rate at which adjacent flow diverges along an axis oriented normal to the flow at the point in question; the opposite of confluence."@en .


<http://sweetontology.net/propDiffusivity/EddyDiffusivity> rdfs:comment "The exchange coefficient for the diffusion of a conservative property by eddies in a turbulent flow."@en .


<http://sweetontology.net""",
            """<http://sweetontology.net/propDiffusivity/Confluence> rdfs:comment "The rate at which adjacent flow is converging along an axis oriented normal to the flow at the point in question."@en .


<http://sweetontology.net/propDiffusivity/Diffluence> rdfs:comment "The rate at which adjacent flow diverges along an axis oriented normal to the flow at the point in question; the opposite of confluence."@en .


<http://sweetontology.net/propDiffusivity/EddyDiffusivity> rdfs:comment "The exchange coefficient for the diffusion of a conservative property by eddies in a turbulent flow."@en .


<http://sweetontology.net""",
            """###  http://purl.obolibrary.org/obo/IAO_0000412
obo1:IAO_0000412 rdf:type owl:AnnotationProperty ;
                 obo1:IAO_0000111 "imported from"@en ;
                 obo1:IAO_0000114 obo1:IAO_0000125 ;
                 obo1:IAO_0000115 "For external terms/classes, the ontology from which the term was imported"@en ;
                 obo1:IAO_0000117 "PERSON:Alan Ruttenberg"@en ,
                                  "PERSON:Melanie Courtot"@en ;
                 obo1:IAO_0000119 "GROUP:""",
            """###  http://purl.obolibrary.org/obo/IAO_0000412
obo1:IAO_0000412 rdf:type owl:AnnotationProperty ;
                 obo1:IAO_0000111 "imported from"@en ;
                 obo1:IAO_0000114 obo1:IAO_0000125 ;
                 obo1:IAO_0000115 "For external terms/classes, the ontology from which the term was imported"@en ;
                 obo1:IAO_0000117 "PERSON:Alan Ruttenberg"@en ,
                                  "PERSON:Melanie Courtot"@en ;
                 obo1:IAO_0000119 "GROUP:""",
            """###  http://purl.obolibrary.org/obo/IAO_0000412
obo1:IAO_0000412 rdf:type owl:AnnotationProperty ;
                 obo1:IAO_0000111 "imported from"@en ;
                 obo1:IAO_0000114 obo1:IAO_0000125 ;
                 obo1:IAO_0000115 "For external terms/classes, the ontology from which the term was imported"@en ;
                 obo1:IAO_0000117 "PERSON:Alan Ruttenberg"@en ,
                                  "PERSON:Melanie Courtot"@en ;
                 obo1:IAO_0000119 "GROUP:""",
            """###  http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym
oboInOwl:hasBroadSynonym rdf:type owl:AnnotationProperty .


###  http://www.geneontology.org/formats/oboInOwl#hasDbXref
oboInOwl:hasDbXref rdf:type owl:AnnotationProperty .


###  http://www.geneontology.org/formats/oboInOwl#hasExactSynonym
oboInOwl:hasExactSynonym rdf:type owl:AnnotationProperty .


###  http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym
oboInOwl:hasNarrowSynonym""",
            """###  http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym
oboInOwl:hasBroadSynonym rdf:type owl:AnnotationProperty .


###  http://www.geneontology.org/formats/oboInOwl#hasDbXref
oboInOwl:hasDbXref rdf:type owl:AnnotationProperty .


###  http://www.geneontology.org/formats/oboInOwl#hasExactSynonym
oboInOwl:hasExactSynonym rdf:type owl:AnnotationProperty .


###  http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym
oboInOwl:hasNarrowSynonym""",
            """###  http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym
oboInOwl:hasBroadSynonym rdf:type owl:AnnotationProperty .


###  http://www.geneontology.org/formats/oboInOwl#hasDbXref
oboInOwl:hasDbXref rdf:type owl:AnnotationProperty .


###  http://www.geneontology.org/formats/oboInOwl#hasExactSynonym
oboInOwl:hasExactSynonym rdf:type owl:AnnotationProperty .


###  http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym
oboInOwl:hasNarrowSynonym""",
            """obo1:BFO_0000055 rdf:type owl:ObjectProperty ;
                 rdfs:domain obo1:BFO_0000015 ;
                 rdfs:range obo1:BFO_0000017 ;
                 obo1:IAO_0000111 "realizes"@en ;
                 obo1:IAO_0000112 "this disease course realizes this disease"@en ,
                                  "this investigation realizes this investigator role"@en ,
                                  "this shattering realizes this fragility"@en ;
                 obo1:IAO_0000600 "to say that b realizes c at t is to assert that there is some material entity d & b is a process which has participant d at t & c is """,
            """obo1:BFO_0000055 rdf:type owl:ObjectProperty ;
                 rdfs:domain obo1:BFO_0000015 ;
                 rdfs:range obo1:BFO_0000017 ;
                 obo1:IAO_0000111 "realizes"@en ;
                 obo1:IAO_0000112 "this disease course realizes this disease"@en ,
                                  "this investigation realizes this investigator role"@en ,
                                  "this shattering realizes this fragility"@en ;
                 obo1:IAO_0000600 "to say that b realizes c at t is to assert that there is some material entity d & b is a process which has participant d at t & c is """,
            """obo1:BFO_0000055 rdf:type owl:ObjectProperty ;
                 rdfs:domain obo1:BFO_0000015 ;
                 rdfs:range obo1:BFO_0000017 ;
                 obo1:IAO_0000111 "realizes"@en ;
                 obo1:IAO_0000112 "this disease course realizes this disease"@en ,
                                  "this investigation realizes this investigator role"@en ,
                                  "this shattering realizes this fragility"@en ;
                 obo1:IAO_0000600 "to say that b realizes c at t is to assert that there is some material entity d & b is a process which has participant d at t & c is """]

import json

# Archivo JSON donde se almacenarán los resultados
filename = 'generated_texts_test_df_calidad_alta_e4_150_to_450_p1.json'

# Intentar cargar el archivo existente si ya contiene datos
try:
    with open(filename, 'r') as file:
        results = json.load(file)
except FileNotFoundError:
    results = {}
c=0
for prompt in prompts_list:
    # Especifica la ruta del modelo local y el prompt
    local_model_path = "/workspace/NAS/GPLSI/llm-train-tokenizer-custom-dataset-main/modelos/Llama-3.2-1B_df_calidad_alta/epoch_4"

    # Llama a la función y muestra el resultado
    generated_text = load_model_and_generate(local_model_path, prompt)
    results[f'{prompt}_{c}'] = generated_text
    print("Texto generado:", generated_text)
    # Guardar los resultados actualizados en el archivo JSON
    with open(filename, 'w') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    c=c+1















import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_generate(model_path, prompt, gpu_id=4, max_length=450):
    """Loads a model and generates text based on the given prompt."""
    
    # Check if the specified GPU is available
    if torch.cuda.device_count() > gpu_id and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU {gpu_id} for text generation.")
    else:
        device = torch.device("cpu")
        print(f"GPU {gpu_id} is not available. Using CPU.")

    # Load the tokenizer and model from the local path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    # Tokenize the prompt and move it to the selected device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate text from the prompt
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)

    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Load prompts from an external JSON file
prompts_file = "prompts.json"

try:
    with open(prompts_file, 'r', encoding='utf-8') as file:
        prompts_list = json.load(file)
except FileNotFoundError:
    print(f"Error: {prompts_file} not found.")
    prompts_list = []
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON in {prompts_file}.")
    prompts_list = []

# List of model paths
models_list = [
    "model_1_path",
    "model_2_path"
]

# Output JSON file
output_file = 'generated_texts_all_models.json'

# Try loading existing results if the file already contains data
try:
    with open(output_file, 'r', encoding='utf-8') as file:
        results = json.load(file)
except FileNotFoundError:
    results = {}
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON in {output_file}. Starting with an empty results dictionary.")
    results = {}

# Generate text for each model
for model_path in models_list:
    model_name = model_path.split("/")[-2]  # Extract model name from path

    if model_name not in results:
        results[model_name] = {}

    # Generate text for each prompt
    for idx, prompt in enumerate(prompts_list):
        generated_text = load_model_and_generate(model_path, prompt)
        results[model_name][f'prompt_{idx}'] = generated_text
        print(f"Generated text for {model_name}, prompt {idx}.")

    # Save updated results to the JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

print(f"All outputs saved in {output_file}.")

