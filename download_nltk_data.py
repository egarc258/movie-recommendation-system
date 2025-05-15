import nltk

print("Downloading NLTK resources...")
# Download all required NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')  # Specific version needed
print("NLTK resources downloaded successfully!")

# Verify the resources are properly downloaded
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    # Test tokenization
    test_sentence = "This is a test sentence for NLTK tokenization."
    tokens = word_tokenize(test_sentence)
    print("Tokenization test:", tokens)
    
    # Test stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    print("Stopwords filtering test:", filtered_tokens)
    
    print("All NLTK resources verified and working correctly!")
except Exception as e:
    print("Error testing NLTK resources:", e)
