# CyberEto \ The Leaky Device

 *Breaking AES-128 ECB with Correlation Power Analysis (CPA)*
## Story Time

So we intercepted a weird little encryption device. Like, physically.
Someone in the team literally stuffed it in the bag and brought it to the lab.

After poking it for a bit, two things became obvious:
- it uses AES-128 in ECB mode (of course it does)
- it leaks power like a faucet

Keep the power thing in mind, as we'll get back to it shortly. 

After hooking an oscilloscope to our device, it was fed 5,000 plaintexts, and the oscilloscope recorded the power traces for each encryption process, and dumped the results in a `traces.npy`.

So we're expected to make something out of this and figure out the key this device used, as it's being implied that this device uses the same key each time it does an encryption. We figure out the key, we get the flag. That's it. 

With that being said, we have three files attached:
- `flag_ciphertext.bin`
- `plaintexts.npy`
- `traces.npy`

## Understanding the Attack

Before we jump into breaking the device, we need to understand our theory. 

Taking a look at the `traces.npy` file, we find out that the traces that we are given are in the form of a 5000x2000 matrix, where each element is a given bool.
```py
import  numpy  as  np

arr  =  np.load("traces.npy")
print(arr.shape) # prints (5000, 2000)
print(arr.dtype) # prints float32
```
So for each plaintext our device encrypted, it consumed electricity, and we have 2,000 samples of that consumption recorded as floats for each one of them. 

Now our problem looks a bit more clear, we need to find a way to exploit these trace samples to figure out what key is being used for the AES. 

Whether you've already seen a problem like this before (or did your research mid-ctf like me), a topic called [*Correlation Power Analysis*](http://wiki.newae.com/Correlation_Power_Analysis) comes up. 

The core idea behind the attack is that AES leaks information during the first round when it computes:

$$
\mathit{SBOX}(\mathit{plaintext\_byte} \oplus \mathit{key\_byte})
$$

Real chips consume slightly more or less power depending on how many set bits  are in the value they’re processing (this is [*the Hamming-Weight model*](https://en.wikipedia.org/wiki/Hamming_weight)). The S-box output has a different Hamming Weight for every possible key guess, which means every guess implies a  _different hypothetical power consumption_.

Since our dataset contains 5,000 plaintexts and their corresponding power traces, we can test every possible value (0–255) for each key byte. For each guess, we predict the S-box output, then convert it to a hypothetical power leak and then measure how strongly it correlates with the real traces. The key guess with the strongest correlation is the correct one.

Normally, brute-forcing AES-128 is $2^{128}$ operations, which is basically longer than the universe has left to live. But thanks to this side-channel leakage, we reduce it to  $256 \times 16 = 4096$. AES itself isn’t the problem, the sloppy implementation of it here is.

## Implementing the Attack

As we said before, we want to brute-force all 256 values for each key-byte position and compare them to the traces we have. The key-byte guess with the highest correlation is the correct one for that position.

For each key-byte position  $j$, we try every key-guess  $k$. For every plaintext/trace index  $i∈[1,N]$, we compute the predicted leakage value under the Hamming-Weight model. This gives us a prediction vector of size  $N$, one value per trace.

Each trace has  $T$  sample points, so for each column  $t∈[1,T]$  in the trace matrix  $trace[i][t]$, we compute the correlation between our prediction vector and that column. For each key-guess  $k$, we record the highest correlation across all  $T$  columns. The key-guess with the highest peak correlation is the correct value for that byte position, and then we move on to the next byte.

That sounds good but how do we compute the correlation value for a byte-guess $k$? 

Given that our current prediction vector is:
 $$P =
\begin{bmatrix}
P_1 \\
P_2 \\
\vdots \\
P_N
\end{bmatrix}$$ and our current trace column is:
 $$C_t =
\begin{bmatrix}
C_{1,t} \\
C_{2,t} \\
\vdots \\
C_{N,t}
\end{bmatrix}$$we compute $\rho(P, C_t) = \mathrm{corr}(P, C_t)$. 

This is given by the following formula: 

$$\mathrm{corr}(P, C_t)=
\frac{\sum_{i=1}^N (P_i - \mu_P)(C_{i,t} - \mu_{C_t})}
     {\sqrt{\sum_{i=1}^N (P_i - \mu_P)^2}\;
      \sqrt{\sum_{i=1}^N (C_{i,t} - \mu_{C_t})^2}}$$

where we have the mean of $P$ and $C_t$
‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ $\mu_P = \frac{1}{N}\sum_{i=1}^N P_i$,   ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ $\mu_{C_t} = \frac{1}{N}\sum_{i=1}^N C_{i,t}$

‎ ‎ ‎ 
Here is how we implement this is in python:
```py
import  numpy  as  np
from  Crypto.Cipher  import  AES

# Load Data
pts  =  np.load("plaintexts.npy")
trc  =  np.load("traces.npy")
n_trc,  n_samp  =  trc.shape

# Sbox & HW precomputation
sbox  =  [
0x63,  0x7c,  0x77,  0x7b,  0xf2,  0x6b,  0x6f,  0xc5,  0x30,  0x01,  0x67,  0x2b,  0xfe,  0xd7,  0xab,  0x76,
0xca,  0x82,  0xc9,  0x7d,  0xfa,  0x59,  0x47,  0xf0,  0xad,  0xd4,  0xa2,  0xaf,  0x9c,  0xa4,  0x72,  0xc0,
0xb7,  0xfd,  0x93,  0x26,  0x36,  0x3f,  0xf7,  0xcc,  0x34,  0xa5,  0xe5,  0xf1,  0x71,  0xd8,  0x31,  0x15,
0x04,  0xc7,  0x23,  0xc3,  0x18,  0x96,  0x05,  0x9a,  0x07,  0x12,  0x80,  0xe2,  0xeb,  0x27,  0xb2,  0x75,
0x09,  0x83,  0x2c,  0x1a,  0x1b,  0x6e,  0x5a,  0xa0,  0x52,  0x3b,  0xd6,  0xb3,  0x29,  0xe3,  0x2f,  0x84,
0x53,  0xd1,  0x00,  0xed,  0x20,  0xfc,  0xb1,  0x5b,  0x6a,  0xcb,  0xbe,  0x39,  0x4a,  0x4c,  0x58,  0xcf,
0xd0,  0xef,  0xaa,  0xfb,  0x43,  0x4d,  0x33,  0x85,  0x45,  0xf9,  0x02,  0x7f,  0x50,  0x3c,  0x9f,  0xa8,
0x51,  0xa3,  0x40,  0x8f,  0x92,  0x9d,  0x38,  0xf5,  0xbc,  0xb6,  0xda,  0x21,  0x10,  0xff,  0xf3,  0xd2,
0xcd,  0x0c,  0x13,  0xec,  0x5f,  0x97,  0x44,  0x17,  0xc4,  0xa7,  0x7e,  0x3d,  0x64,  0x5d,  0x19,  0x73,
0x60,  0x81,  0x4f,  0xdc,  0x22,  0x2a,  0x90,  0x88,  0x46,  0xee,  0xb8,  0x14,  0xde,  0x5e,  0x0b,  0xdb,
0xe0,  0x32,  0x3a,  0x0a,  0x49,  0x06,  0x24,  0x5c,  0xc2,  0xd3,  0xac,  0x62,  0x91,  0x95,  0xe4,  0x79,
0xe7,  0xc8,  0x37,  0x6d,  0x8d,  0xd5,  0x4e,  0xa9,  0x6c,  0x56,  0xf4,  0xea,  0x65,  0x7a,  0xae,  0x08,
0xba,  0x78,  0x25,  0x2e,  0x1c,  0xa6,  0xb4,  0xc6,  0xe8,  0xdd,  0x74,  0x1f,  0x4b,  0xbd,  0x8b,  0x8a,
0x70,  0x3e,  0xb5,  0x66,  0x48,  0x03,  0xf6,  0x0e,  0x61,  0x35,  0x57,  0xb9,  0x86,  0xc1,  0x1d,  0x9e,
0xe1,  0xf8,  0x98,  0x11,  0x69,  0xd9,  0x8e,  0x94,  0x9b,  0x1e,  0x87,  0xe9,  0xce,  0x55,  0x28,  0xdf,
0x8c,  0xa1,  0x89,  0x0d,  0xbf,  0xe6,  0x42,  0x68,  0x41,  0x99,  0x2d,  0x0f,  0xb0,  0x54,  0xbb,  0x16
]
hw  =  [bin(x).count('1')  for  x  in  range(256)] 

key  =  []

# Pre-calculate trace stats
# We just need the mean of each column to do the math
t_means  =  np.mean(trc,  axis=0)
t_devs  =  trc  -  t_means
t_vars  =  np.sum(t_devs**2,  axis=0)

print("Starting process..")

for  b  in  range(16):
	max_c  =  0
	best_k  =  0
	p_col  =  pts[:,  b]

	for  k  in  range(256):

		# Generate Prediction Vector (Size N)
		h  =  [hw[sbox[p  ^  k]]  for  p  in  p_col]
		
		# Manual correlation math
		h_mean  =  np.mean(h)
		h_dev  =  h  -  h_mean
		h_var  =  np.sum(h_dev**2)
		
		# Walk T times (check against every instance)
		for  t  in  range(n_samp):
		
		# Extract column T
		t_col_dev  =  t_devs[:,  t]

		# Pearson Correlation Formula
		cov  =  np.dot(h_dev,  t_col_dev)
		corr  =  cov  /  np.sqrt(h_var  *  t_vars[t])

		if  abs(corr)  >  max_c:
			max_c  =  abs(corr)
			best_k  =  k

	key.append(best_k)
	print(f"Byte {b}: {best_k:02x}")

final_key  =  bytes(key)
print(f"Key: {final_key.hex()}")
```
After running this solver script, we'll get the key `f6ea854dd3b268a272bc1b1d2937c27b`. Having known this, we can finally decrypt our flag:
```py
from  Crypto.Cipher  import  AES  

key_hex  =  "f6ea854dd3b268a272bc1b1d2937c27b"
final_key  =  bytes.fromhex(key_hex)

with  open("flag_ciphertext.bin",  "rb")  as  f:
	ciphertext  =  f.read()

  
# Decrypt
cipher  =  AES.new(final_key,  AES.MODE_ECB)
flag  =  cipher.decrypt(ciphertext)

print(flag.decode('utf-8'))
```
And there we have it, our flag is `cybereto{SRJ_OK}`.

## Final Thoughts
Overall, the challenge demonstrates how little leakage is needed to break AES when the implementation isn’t protected. Even though the AES algorithm itself is secure, the traces give away enough information to recover the key one byte at a time using simple statistical analysis. The process is repetitive, but once the pipeline is set up, each key-byte falls quickly. The main lesson is that side-channel attacks exploit the _implementation_, not the math, and even a small amount of unprotected leakage is enough to fully recover the secret key.

##### [Check out the challenge files](https://github.com/mrshmelloww/crypto-writeups/tree/main/cybereto/theLeakyDevice/challengeFiles)
