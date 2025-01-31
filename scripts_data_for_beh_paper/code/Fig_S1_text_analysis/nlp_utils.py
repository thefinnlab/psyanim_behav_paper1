import os, sys
import numpy as np
import pandas as pd
import regex as re
import torch

def load_mlm_model(model_name, cache_dir=None):
	'''
	Use a model from the sentence-transformers library to get
	sentence embeddings. Models used are trained on a next-sentence
	prediction task and evaluate the likelihood of S2 following S1.
	'''
	# set the path of where to download models
	# this NEEDS to be run before loading from transformers
	if cache_dir:
		os.environ['TRANSFORMERS_CACHE'] = cache_dir

	from transformers import AutoTokenizer, AutoModel

	# Load model from HuggingFace Hub
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	model = AutoModel.from_pretrained(model_name)
	
	model.eval()

	return tokenizer, model

import re

def subwords_to_words(sentence, tokenizer):
	
	word_token_pairs = []
	
	# split the sentence on spaces + punctuation (excluding apostrophes and hyphens within words)
	regex_split_pattern = r'(\w|\.\w|\:\w|\'\w|\'\w|\-\w|\S)+'

	# regex_split_pattern = r"[\w]+[''.-:]?[\w]*"

	for m in re.finditer(regex_split_pattern, sentence):
		word = m.group(0)
		tokens = tokenizer.encode(word, add_special_tokens=False)
		char_idxs = (m.start(), m.end()-1)
		
		word_token_pairs.append((word, tokens, char_idxs))
	
	return word_token_pairs

def extract_word_embeddings(sentences, tokenizer, model):#, word_indices=None):
	'''
	Given a list of sentences, pass them through the tokenizer/model. Then pair
	sub-word tokens into the words of the actual sentence and extract the true
	word embeddings. 
	
	If wanted, can return only certain indices (specified by word_indices)
	
	Currently not robust to different length strings MBMB
	'''
	
	if isinstance(sentences, str):
		sentences = [sentences]
	
	if not sentences:
		return []
	
	# get the full sentence tokenized
	encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
	
	# get the embeddings
	with torch.no_grad():
		model_output = model(**encoded_inputs, output_hidden_states=True)
	
	all_embeddings = []
	
	# bring together the current sentence, its tokens, and its embeddings
	for i, sent in enumerate(sentences):
		# now pair subwords into words for the current sentence
		subword_word_pairs = subwords_to_words(sent, tokenizer)
		print(sent, subword_word_pairs)
		
		embeddings = []
		
		# for the current set of word subword pairs, get the embeddings
		for (word, tokens, char_span) in subword_word_pairs:
			
			# given the character to token mapping in the sentence, 
			# find the first and last token indices
			start_token = encoded_inputs.char_to_token(batch_or_char_index=i, char_index=char_span[0])
			end_token = encoded_inputs.char_to_token(batch_or_char_index=i, char_index=char_span[-1])
			
			# extract the embedding for the given word
			word_embed = torch.stack([layer[i, start_token:end_token+1, :].sum(0) for layer in model_output['hidden_states']])
			embeddings.append(word_embed)
			
		# stack the embeddings together
		embeddings = torch.stack(embeddings)
		
		# make sure the mapping happened correctly
		if len(sent.split()) != embeddings.shape[0]:
			print (subword_word_pairs)
			print (len(subword_word_pairs))
			print (embeddings.shape)
			print (len(sent.split()))

		assert (len(sent.split()) == embeddings.shape[0])
		
		all_embeddings.append(embeddings)
	
	# all_embeddings = torch.stack(all_embeddings)
	
	# if word_indices:
	# 	return all_embeddings[:, word_indices, :]
	# else:
	# 	return all_embeddings

	return all_embeddings