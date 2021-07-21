from util_mmtf import *
from collections import defaultdict
import json

from ProteinPairsGenerator.PreProcessing import readPotein

MAX_LENGTH = 500

# CATH base URL
cath_base_url = "http://download.cathdb.info/cath/releases/all-releases/v4_2_0/"
cath_base_url = "http://download.cathdb.info/cath/releases/latest-release/"

# CATH hierarchical classification
cath_domain_fn = "cath-domain-list.txt"
cath_domain_url = cath_base_url + "cath-classification-data/" + cath_domain_fn
cath_domain_file = "cath/cath-domain-list.txt"
download_cached(cath_domain_url, cath_domain_file)

# CATH topologies
cath_nodes = defaultdict(list)
with open(cath_domain_file,"r") as f:
    lines = [line.strip() for line in f if not line.startswith("#")]
    for line in lines:
        entries = line.split()
        cath_id, cath_node = entries[0], ".".join(entries[1:4])
        chain_name = cath_id[:4] + "." + cath_id[4]
        cath_nodes[chain_name].append(cath_node)
cath_nodes = {key:list(set(val)) for key,val in cath_nodes.items()}

for datasetType in ["training_100"]: # "validation"
    dataset = []
    tooLong, badFormat = 0, 0
    with open("../proteinNetNew/raw/casp12/" + datasetType) as inFile:
        while True:
            x = readPotein(inFile)

            if x is None:
                break

            try:
                valid = x["id"].split("#")[-1]
                print(valid)
                pdb, _, chain = valid.split("_")
                assert len(pdb) == 4 and len(chain) == 1
            except  Exception:
                print("WUT")
                continue

            try:

                chain_dict = mmtf_parse(pdb, chain)

                if len(chain_dict["seq"]) <= MAX_LENGTH:
                    chain_name = pdb.lower() + "." + chain
                    chain_dict["name"] = chain_name
                    chain_dict["CATH"] = cath_nodes[chain_name]
                    dataset.append(chain_dict)
                else:
                    tooLong += 1
            except Exception as e:
                badFormat += 1

    print("Too long: {}\nBad format: {}\nSize dataset: {}".format(tooLong, badFormat, len(dataset)))
    outfile = datasetType +".jsonl"
    with open(outfile, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
