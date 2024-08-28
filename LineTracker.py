"""
Description : File for unifying all line tracker classes and methods. 
Author      : Vithor Bertalan
Version     : Mar 25, 2024
"""

## Imports libraries and paths

import linetracker.embeddings.llm as llm_embedding
import linetracker.parser.variables_matrix as ev
import linetracker.clustering.kmedoid as clustK
import linetracker.embeddings.distances as d
import linetracker.line_distance as ld
import linetracker.parser.parser as p
import linetracker.main as m

from rich.console import Console
from tqdm import tqdm
import numpy as np
import colorsys
import h5py
import json
import time
import sys
import os

## Defining class 

class LineTracker:
    """ Class object to run the methods inside the API
    """

    def __init__(self):
        self.logfile = 0
        self.token = 0
        self.num_clusters = 0
        self.embedder = 0
        self.embedding_distance_fn = 0
        self.line_distance_fn = 0
        self.clustering_fn = 0
        self.float_precision = 0
        self.triplet_coefficient = 0
        self.dict_parsed_variables = {}
        self.splits_samples = {}
        self.parser = 0
        self.log_size = 0
        self.dict_variables_distance_matrix = {}
        self.dict_embeddings_distance_matrix = {}
        self.dict_count_matrix = {}
        self.dico_combined_matrix = {}
        self.dico_clustering_output = {}

    # Method to run the pipeline of clustering.
    def cluster(self, logfile, token=""):
        self.load_file(logfile)
        self.token = token
        self.build_functions()
        self.parses_variables()
        self.builds_variable_matrix()
        self.builds_embedding_matrix()
        self.builds_count_matrix()
        self.builds_combined_matrix()
        self.clusters_matrix()
        return(self.build_html())

    # Method to load the file. If JSON, file is read. If raw text, the content is converted to JSON.
    def load_file(self, logfile):
        ext = os.path.splitext(logfile)[-1].lower()
        if ext == ".json":
            with open(logfile) as fp:
                print("Reading JSON file.")
                splits_samples = json.load(fp)
                self.splits_samples = {k:v for k,v in sorted(splits_samples.items(),key=lambda x:int(x[0]))}
                self.log_size = int(list(self.splits_samples.keys())[0])
                self.logfile = logfile
                return(True)
        else:
            print ("Reading raw text file.")            
            splits_samples = self.json_converter(logfile)
            self.splits_samples = {k:v for k,v in sorted(splits_samples.items(),key=lambda x:int(x[0]))}
            self.log_size = int(list(self.splits_samples.keys())[0])
            self.logfile = logfile
            return(True)
    
    # Auxiliary method to convert raw file to JSON. 
    def json_converter(self,logfile):
        list_of_dicts = [] 
        with open(logfile) as fh:
            for i, line in enumerate(fh): 
                current_dict = {'line_num':i, 'text':line}
                list_of_dicts.append(current_dict) 

        result_dict = {i: [logfile, list_of_dicts]}
        return (result_dict)        
    
    # Method to call the parsing and embedding metafunctions. 
    def build_functions(self):
         # Parsing
        self.parser = lambda logs:p.get_parsing_drainparser([e['text'] for e in logs],reg_expressions=[r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"],depth=3,similarity_threshold=0.1,max_children=5)
        # Creating embeddings
        models_names = ["meta-llama/Llama-2-7b-chat-hf","WhereIsAI/UAE-Large-V1", "BAAI/bge-large-en-v1.5"]
        model_name = models_names[2]
        init_embedder = llm_embedding.generate_embeddings_llm(model_name=model_name,token = self.token,use_cpu=True)
        # Using pooling by mean
        pooling_fn = llm_embedding.get_pooling_function("mean")
        # Setting coefficients
        self.embedder = lambda logs: init_embedder(logs, pooling_fn,limit_tokens=100,precision=np.float16)# type: ignore
        self.embedding_distance_fn = d.normalized_cosine_distance
        self.line_distance_fn = ld.get_absolute_line_distance_matrix
        self.float_precision = np.float16
        self.triplet_coefficient = m.TripletCoef(coef_variables_matrix=0.0, coef_embeddings_matrix=1, coef_count_matrix=0.0)

    # Parses variables using Drain. 
    def parses_variables(self):        
        # for each of our log file with different number of lines, we apply step 1 and save the result
        for size,[build_log_name, logs] in self.splits_samples.items():
            parsed_logs = self.parser(logs)
            # Apply step 1
            logs_texts = [e['text'] for e in logs]
            parsed_variables = [e['variables'] for e in parsed_logs]
            # And save the result
            self.dict_parsed_variables[size] = parsed_variables

    # Builds variable matrix, using parsed tokens. 
    def builds_variable_matrix(self):
        for size,parsed_variables in self.dict_parsed_variables.items():
            variables_distance_matrix = ev.get_variable_matrix(parsed_events=parsed_variables)
            self.dict_variables_distance_matrix[size] = variables_distance_matrix

    # Builds embedding matrix, using LLM.
    def builds_embedding_matrix(self):
        for size,[build_log_name, logs] in self.splits_samples.items():
            logs_texts = [e['text'] for e in logs]
            start = time.perf_counter()
            embeddings = np.array(
                [embedding for embedding in tqdm(self.embedder(logs_texts), total=self.log_size)]
            ).astype(self.float_precision)
            diff = time.perf_counter()-start
            embeddings_distance_matrix = self.embedding_distance_fn(embeddings).astype(
                self.float_precision
            )
            self.dict_embeddings_distance_matrix[size] = embeddings_distance_matrix
            del embeddings

    # Builds count matrix. 
    def builds_count_matrix(self):
        for size,[build_log_name, logs] in self.splits_samples.items():
            count_matrix = self.line_distance_fn(logs).astype(self.float_precision)
            self.dict_count_matrix[size] = count_matrix

    # Builds combined matrix, considering alpha, beta and gamma. 
    def builds_combined_matrix(self):        
        for size in self.dict_count_matrix:
            for i,mat in enumerate([self.dict_variables_distance_matrix[size],self.dict_embeddings_distance_matrix[size],self.dict_count_matrix[size]]):
                assert np.unique(np.diag(mat)).tolist() == [0], f"Error for matrix {i}\n{mat}"
            self.dico_combined_matrix[size] = m.combine_matrices(
                m.TripletMatrix(
                    variables_matrix=self.dict_variables_distance_matrix[size],
                    embeddings_matrix=self.dict_embeddings_distance_matrix[size],
                    count_matrix=self.dict_count_matrix[size],
                ),
                triplet_coef=self.triplet_coefficient,
            ).astype(self.float_precision)

    # Clusters the resulting matrix
    def clusters_matrix(self,must_link=[],cannot_link=[]):
        self.clustering_fn = lambda combined_matrix: clustK.get_clustering_kmedoid(combined_matrix, number_of_clusters=self.num_clusters, must_link = must_link, cannot_link=cannot_link)[0]['clustering']     
        for size,matrix in self.dico_combined_matrix.items():
            self.dico_clustering_output[size] = self.clustering_fn(matrix)

    # Auxiliary method to create color palettes. 
    def generate_hsv_palette(self, num_colors, saturation=1.0, value=1.0):
        colors = []
        hue_step = 1.0 / num_colors
        for i in range(num_colors):
            hue = i * hue_step
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            rgb = tuple(int(x * 255) for x in rgb)
            colors.append(rgb)
        return colors        

    # Builds HTML from clustering results. 
    def build_html(self):
        console = Console(color_system="auto", record=True)
        self.dico_clustering_output = {s:self.dico_clustering_output[s] for s in sorted(self.dico_clustering_output,key=lambda x:int(x))}
        for size, _ in self.dico_clustering_output.items():
            console.log(f"{size:-^100}\n", style=f"white", end="" )
            clustering_output = list(self.dico_clustering_output[size].values())
            unique_clusters =  np.unique(clustering_output)
            mapping = {clust:col for clust,col in zip(unique_clusters,self.generate_hsv_palette(len(unique_clusters),saturation=0.75))}
            for line_id, (log,cluster) in enumerate(zip(self.splits_samples[size][1], clustering_output)):
                text = f"{line_id:03d}-{cluster}: {log['text']}"
                r,g,b = mapping[cluster]
                console.log(text, style=f"rgb({r},{g},{b})", end="" )

        format = code_format='<!DOCTYPE html>\n<html>\n<head>\n<meta charset="UTF-8">\n<style>\n{stylesheet}\nbody {{\n    color: {foreground};\n    background-color: #000000;\n}}\n</style>\n</head>\n<body>\n    <pre style="font-family:Menlo,\'DejaVu Sans Mono\',consolas,\'Courier New\',monospace"><code>{code}</code></pre>\n</body>\n</html>\n'
        # To save the HTML locally
        # console.save_html(self.logfile + ".html", code_format=format)
        # To export the HTML in the API
        html = console.export_html(code_format=format)
        return(html)