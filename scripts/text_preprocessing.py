import re

# Intelligent text processing functions
def sent_tokenize(text):
    """Split text into sentences using regex patterns."""
    # Match sentence endings with punctuation followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def word_tokenize(text):
    """Split text into words."""
    # Split on whitespace and remove punctuation
    words = re.findall(r'\b\w+\b', text.lower())
    return words

# Load stopwords from file
def load_stopwords(filepath="stopwords.txt"):
    """Load stopwords from a file into a set for O(1) lookup."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Warning: Stopwords file '{filepath}' not found. Using empty stopwords set.")
        return set()

def rule_based_lemmatize(word):
    """
    Apply rule-based lemmatization based on common English patterns.
    This function identifies common suffixes and transforms them to base forms.
    """
    word = word.lower()
    
    # Handle irregular plurals first
    irregular_plurals = {
        "men": "man", "women": "woman", "children": "child", "people": "person",
        "mice": "mouse", "feet": "foot", "teeth": "tooth", "geese": "goose",
        "oxen": "ox", "phenomena": "phenomenon", "criteria": "criterion",
        "data": "datum", "analyses": "analysis", "diagnoses": "diagnosis",
        "theses": "thesis", "crises": "crisis", "hypotheses": "hypothesis"
    }
    
    if word in irregular_plurals:
        return irregular_plurals[word]
    
    # Handle some common irregular verbs
    irregular_verbs = {
        "am": "be", "is": "be", "are": "be", "was": "be", "were": "be", "been": "be", "being": "be",
        "has": "have", "had": "have", "having": "have",
        "does": "do", "did": "do", "doing": "do",
        "goes": "go", "went": "go", "gone": "go", "going": "go",
        "makes": "make", "made": "make", "making": "make",
        "says": "say", "said": "say", "saying": "say",
        "sees": "see", "saw": "see", "seen": "see", "seeing": "see",
        "takes": "take", "took": "take", "taken": "take", "taking": "take"
    }
    
    if word in irregular_verbs:
        return irregular_verbs[word]
    
    # Rule 1: -ing endings (for verbs)
    if word.endswith('ing'):
        # Double letter + ing: running -> run
        if len(word) > 4 and word[-4] == word[-5]:
            return word[:-4]
        # e + ing: hiking -> hike
        elif word.endswith('eing'):
            return word[:-3]
        # ing: working -> work
        else:
            stem = word[:-3]
            if len(stem) > 1:  # Make sure we have a valid stem
                return stem
    
    # Rule 2: -ed endings (for verbs)
    if word.endswith('ed'):
        # Double letter + ed: stopped -> stop
        if len(word) > 3 and word[-3] == word[-4]:
            return word[:-3]
        # e + ed: liked -> like
        elif word.endswith('eed'):
            return word[:-1]
        # ed: worked -> work
        else:
            stem = word[:-2]
            if len(stem) > 1:  # Make sure we have a valid stem
                return stem
    
    # Rule 3: -s endings (plural nouns and 3rd person verbs)
    if word.endswith('s') and not word.endswith('ss'):
        # ies: countries -> country
        if word.endswith('ies'):
            return word[:-3] + 'y'
        # es: boxes -> box
        elif word.endswith('es'):
            if word.endswith('sses'):  # glasses -> glass
                return word[:-2]
            else:
                return word[:-2]
        # s: cats -> cat
        else:
            return word[:-1]
    
    # Rule 4: -er/-est endings (comparative/superlative adjectives)
    if word.endswith('er') and len(word) > 3:
        # Double letter + er: bigger -> big
        if word[-3] == word[-4]:
            return word[:-3]
        # e + er: nicer -> nice
        elif word.endswith('ier'):
            return word[:-3] + 'y'
        else:
            return word[:-2]
    
    if word.endswith('est') and len(word) > 4:
        # Double letter + est: biggest -> big
        if word[-4] == word[-5]:
            return word[:-4]
        # e + est: nicest -> nice
        elif word.endswith('iest'):
            return word[:-4] + 'y'
        else:
            return word[:-3]
    
    # Rule 5: -ly endings (adverbs)
    if word.endswith('ly') and len(word) > 3:
        return word[:-2]
    
    # Rule 6: -ful, -ness, -ment, -ity (noun and adjective transformations)
    if word.endswith('ful') and len(word) > 4:
        return word[:-3]
    
    if word.endswith('ness') and len(word) > 5:
        stem = word[:-4]
        if stem.endswith('i'):
            return stem[:-1] + 'y'  # happiness -> happy
        return stem
    
    if word.endswith('ment') and len(word) > 5:
        return word[:-4]
    
    if word.endswith('ity') and len(word) > 4:
        return word[:-3] + 'e'  # activity -> active
    
    # Return the word unchanged if no rules apply
    return word

def preprocess_text(text, stopwords, save_tokens=False):
    """
    Preprocess text by lowercasing, removing non-alphabetic characters,
    tokenizing, removing stopwords, and applying rule-based lemmatization.
    
    If save_tokens is True, returns both the cleaned text and the tokens
    """
    # Convert to lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize into words
    tokens = word_tokenize(text)
    
    # Remove stopwords using the provided set
    filtered_tokens = [word for word in tokens if word not in stopwords]
    
    # Apply rule-based lemmatization
    lemmatized_tokens = [rule_based_lemmatize(token) for token in filtered_tokens]
    
    # Join tokens back into a string
    cleaned_text = ' '.join(lemmatized_tokens)
    
    if save_tokens:
        return cleaned_text, lemmatized_tokens
    return cleaned_text
