import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def precision_at_position_k(data, target, k):
    return data[:k].count(target) / k

def average_precission(data, target):
    if isinstance(data, np.ndarray):
        values = data.tolist()
    else:
        values = data
        
    return sum([precision_at_position_k(values, target, k) 
                for k, value in enumerate(values, start=1) if value == target]) / values.count(target)

def custom_ap_scorer(y_true, y_predicted):
    rank_predict = np.argsort(-1*y_predicted)
    return average_precission(y_true[rank_predict], 1)

sklearn_custom_ap_scorer = make_scorer(custom_ap_scorer)

sem_abrv_to_full = {"aapp": "Amino Acid, Peptide, or Protein", "acab": "Acquired Abnormality", "acty": "Activity", "aggp": "Age Group", "amas": "Amino Acid Sequence", "amph": "Amphibian", "anab": "Anatomical Abnormality", "anim": "Animal", "anst": "Anatomical Structure", "antb": "Antibiotic", "arch": "Archaeon", "bacs": "Biologically Active Substance", "bact": "Bacterium", "bdsu": "Body Substance", "bdsy": "Body System", "bhvr": "Behavior", "biof": "Biologic Function", "bird": "Bird", "blor": "Body Location or Region", "bmod": "Biomedical Occupation or Discipline", "bodm": "Biomedical or Dental Material", "bpoc": "Body Part, Organ, or Organ Component", "bsoj": "Body Space or Junction", "celc": "Cell Component", "celf": "Cell Function", "cell": "Cell", "cgab": "Congenital Abnormality", "chem": "Chemical", "chvf": "Chemical Viewed Functionally", "chvs": "Chemical Viewed Structurally", "clas": "Classification", "clna": "Clinical Attribute", "clnd": "Clinical Drug", "cnce": "Conceptual Entity", "comd": "Cell or Molecular Dysfunction", "crbs": "Carbohydrate Sequence", "diap": "Diagnostic Procedure", "dora": "Daily or Recreational Activity", "drdd": "Drug Delivery Device", "dsyn": "Disease or Syndrome", "edac": "Educational Activity", "eehu": "Environmental Effect of Humans", "elii": "Element, Ion, or Isotope", "emod": "Experimental Model of Disease", "emst": "Embryonic Structure", "enty": "Entity", "enzy": "Enzyme", "euka": "Eukaryote", "evnt": "Event", "famg": "Family Group", "famg": "Family Group", "ffas": "Fully Formed Anatomical Structure", "fish": "Fish", "fndg": "Finding", "fngs": "Fungus", "food": "Food", "ftcn": "Functional Concept", "genf": "Genetic Function", "geoa": "Geographic Area", "gngm": "Gene or Genome", "gora": "Governmental or Regulatory Activity", "grpa": "Group Attribute", "grup": "Group", "hcpp": "Human-caused Phenomenon or Process", "hcro": "Health Care Related Organization", "hlca": "Health Care Activity", "hops": "Hazardous or Poisonous Substance", "horm": "Hormone", "humn": "Human", "idcn": "Idea or Concept", "imft": "Immunologic Factor", "inbe": "Individual Behavior", "inch": "Inorganic Chemical", "inpo": "Injury or Poisoning", "inpr": "Intellectual Product", "irda": "Indicator, Reagent, or Diagnostic Aid", "lang": "Language", "lbpr": "Laboratory Procedure", "lbtr": "Laboratory or Test Result", "mamm": "Mammal", "mbrt": "Molecular Biology Research Technique", "mcha": "Machine Activity", "medd": "Medical Device", "menp": "Mental Process", "mnob": "Manufactured Object", "mobd": "Mental or Behavioral Dysfunction", "moft": "Molecular Function", "mosq": "Molecular Sequence", "neop": "Neoplastic Process", "nnon": "Nucleic Acid, Nucleoside, or Nucleotide", "npop": "Natural Phenomenon or Process", "nusq": "Nucleotide Sequence", "ocac": "Occupational Activity", "ocdi": "Occupation or Discipline", "orch": "Organic Chemical", "orga": "Organism Attribute", "orgf": "Organism Function", "orgm": "Organism", "orgt": "Organization", "ortf": "Organ or Tissue Function", "patf": "Pathologic Function", "phob": "Physical Object", "phpr": "Phenomenon or Process", "phsf": "Physiologic Function", "phsu": "Pharmacologic Substance", "plnt": "Plant", "podg": "Patient or Disabled Group", "popg": "Population Group", "prog": "Professional or Occupational Group", "pros": "Professional Society", "qlco": "Qualitative Concept", "qnco": "Quantitative Concept", "rcpt": "Receptor", "rept": "Reptile", "resa": "Research Activity", "resd": "Research Device", "rnlw": "Regulation or Law", "sbst": "Substance", "shro": "Self-help or Relief Organization", "socb": "Social Behavior", "sosy": "Sign or Symptom", "spco": "Spatial Concept", "tisu": "Tissue", "tmco": "Temporal Concept", "topp": "Therapeutic or Preventive Procedure", "virs": "Virus", "vita": "Vitamin", "vtbt": "Vertebrate"}