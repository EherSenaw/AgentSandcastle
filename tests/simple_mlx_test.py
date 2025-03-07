import mlx.core as mx
import mlx.nn as nn
import math

MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
FLAG_TRUST_REMOTE = True

def print_outputs(outputs):
	for output in outputs:
		prompt = output.prompt
		generated_text = output.outputs[0].text
		print(f"Prompt: {prompt}\n\nGenerated text: {generated_text}")
	print("-"*80)

class LlamaAttention(nn.Module):
	def __init__(self, dims: int, num_heads: int):
		super().__init__()

		self.num_heads = num_heads

		self.rope = nn.RoPE(dims // num_heads, traditional=True)
		self.query_proj = nn.Linear(dims, dims, bias=False)
		self.key_proj = nn.Linear(dims, dims, bias=False)
		self.value_proj = nn.Linear(dims, dims, bias=False)
		self.out_proj = nn.Linear(dims, dims, bias=False)

	def __call__(self, queries, keys, values, mask=None, cache=None):
		queries = self.query_proj(queries)
		keys = self.key_proj(keys)
		values = self.value_proj(values)

		num_heads = self.num_heads
		B, L, D = queries.shape

		queries = queries.reshape(B, L, num_heads, -1).transpose(0,2,1,3)
		keys = keys.reshape(B, L, num_heads, -1).transpose(0,2,1,3)
		values = values.reshape(B, L, num_heads, -1).transpose(0,2,1,3)

		# Add RoPE to the quries and keys and combine them with the cache.
		if cache is not None:
			key_cache, value_cache = cache
			queries = self.rope(queries, offset=key_cache.shape[2])
			keys = self.rope(keys, offset=key_cache.shape[2])
			keys = mx.concatenate([key_cache, keys], axis=2)
			values = mx.concatenate([value_cache, values], axis=2)
		else:
			queries = self.rope(queries)
			keys = self.rope(keys)
		
		scale = math.sqrt(1 / queries.shape[-1])
		scores = (queries * scale) @ keys.transpose(0,1,3,2)
		if mask is not None:
			scores = scores + mask
		scores = mx.softmax(scores, axis=-1)
		values_hat = (scores @ values).transpose(0,2,1,3).reshape(B,L,-1)

		# Note that we return the keys and values to possibly be used as a cache.
		return self.out_proj(values_hat), (keys, values)

class LlamaEncoderLayer(nn.Module):
	def __init__(self, dims: int, mlp_dims: int, num_heads: int):
		super().__init__()

		self.attention = LlamaAttention(dims, num_heads)

		self.norm1 = nn.RMSNorm(dims)
		self.norm2 = nn.RMSNorm(dims)

		self.linear1 = nn.Linear(dims, mlp_dims, bias=False)
		self.linear2 = nn.Linear(dims, mlp_dims, bias=False)
		self.linear3 = nn.Linear(mlp_dims, dims, bias=False)

	def __call__(self, x, mask=None, cache=None):
		y = self.norm1(x)
		y, cache = self.attention(y, y, y, mask, cache)
		x = x + y

		y = self.norm2(x)
		a = self.linear1(y)
		b = self.linear2(y)
		y = a * mx.sigmoid(a) * b
		y = self.linear3(y)
		x = x + y

		return x, cache

class Llama(nn.Module):
	def __init__(self, num_layers: int, vocab_size: int, dims: int, mlp_dims: int, num_heads: int):
		super().__init__()

		self.embedding = nn.Embedding(vocab_size, dims)
		self.layers = [
			LlamaEncoderLayer(dims, mlp_dims, num_heads) for _ in range(num_layers)
		]
		self.norm = nn.RMSNorm(dims)
		self.out_proj = nn.Linear(dims, vocab_size, bias=False)

	def __call__(self, x):
		# NOTE: USE for training, not for generation;
		# 		because this implementation is focused on single input processing
		#		and not utilizing cache.
		mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
		mask = mask.astype(self.embedding.weight.dtype)

		x = self.embedding(x)
		for l in self.layers:
			x, _ = l(x, mask)
		x = self.norm(x)
		return self.out_proj(x)
	
	def generate(self, x, temp=1.0):
		cache = []

		# Make an additive causal mask. We will need that to process the prompt.
		mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
		mask = mask.astype(self.embedding.weight.dtype)

		x = self.embedding(x)
		for l in self.layers:
			x, c = l(x, mask=mask)
			cache.append(c) # we store that per layer cache in a
							# simple python list.

		x = self.norm(x)
		y = self.out_proj(x[:, -1]) # <- we only care about the last logits
									# 	that generate the next token.
		y = mx.random.categorical(y * (1/temp))

		# y now has size [1]
		# Since MLX is lazily evaluated nothing is computed yet.
		# Calling y.item() would force the computation to happen at
		# this point but we can also choose not to do that and let the
		# user choose when to start the computation
		yield y

		# Now we parsed the prompt and generated the first token we
		# need to feed it back into the model and loop to generate the
		# rest.

		while True:
			# Unsqueezing the last dimension to add a sequence length
			# dimension of 1
			x = y[:, None]

			x = self.embedding(x)
			for i in range(len(cache)):
				# We are overwriting the arrays in the cache list. When
				# the computation will happen, MLX will be discarding the
				# old cache the moment it is not needed anymore.
				x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
			x = self.norm(x)
			y = self.out_proj(x[:, -1])
			y = mx.random.categorical(y * (1/temp))

			yield y
		


if __name__ == '__main__':
	print("="*80)

	conversation = [
		{
			"role": "system",
			"content": "You are a helpful assistant"
		},
		{
			"role": "user",
			"content": "hello"
		},
		{
			"role": "assistant",
			"content": "Hello! How can I assist you today?"
		},
		{
			"role": "user",
			"content": "Write an essay about the importance of higher education."
		},
	]
	conversations = [conversation for _ in range(3)]

	MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
	FLAG_TRUST_REMOTE = True
	'''
	llm = LLM(
		model=MODEL_NAME,
		#dtype=torch.bfloat16,
		dtype="half",
		trust_remote_code=FLAG_TRUST_REMOTE,
		#enable_chunked_prefill=True, --> Bfloat16
	)
	outputs = llm.chat(
		conversation,
		sampling_params=sampling_params,
		use_tqdm=False,
	)
	print_outputs(outputs)
	'''

	llm = Llama(num_layers=12, vocab_size=8192, dims=512, mlp_dims=1024, num_heads=8)
	# Lazy evaluation
	mx.eval(llm.parameters())

	prompt = mx.array([[1, 10, 8, 32, 44, 7]])	# <- Note the double brackets because we
												#	have a batch dimension even
												#	though it is 1 in this case
	generated = [t for i, t in zip(range(10), llm.generate(prompt, 0.8))]
	# Lazy evaluation
	#mx.eval(generated)
	print(generated)

	#print_outputs(outputs)
	del llm