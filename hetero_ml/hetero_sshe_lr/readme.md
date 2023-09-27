"SSHE_LR" 是 "Secure Share and Homomorphic Encryption based Logistic Regression" 的简写。

该算法的优点是无需第三方，从而提高了安全性；缺点是...。

该算法的思路借鉴FATE中实现的 "hetero_sshe_lr"。



# 1. Math

## 1.1. Loss

$$
\begin{align}
L&=-y\ln(\sigma(z))-(1-y)\ln(1-\sigma(z))\\
\\
\text{taylor approx:}\qquad L&\approx-\ln\frac{1}{2}+\frac{1}{2}z-yz+\frac{1}{8}z^2
\end{align}
$$

## 1.2. Gradient

$$
\begin{align}
\frac{\partial L}{\partial z}&=\sigma(z)-y\\
\\
\text{taylor approx:}\qquad \frac{\partial L}{\partial z}&\approx\frac{1}{2}+\frac{1}{4}z-y
\end{align}
$$

# 2. Algorithms

## 2.1. Fit

$$
\begin{array}{lll}
\qquad\qquad\qquad\qquad\qquad & \text{A(Host)}\qquad\qquad\qquad\qquad\qquad\qquad\qquad & \text{B(Guest)}\qquad\qquad\qquad\qquad\qquad\qquad\qquad\\
\text{share\_model} & w_{A,[A]},\ w_{B,[A]} & w_{A,[B]},\ w_{B,[B]}\\
\text{cal\_prediction} & y_{[A]} & [[z]]^A,\ y_{[B]}\\
& e_{[A]}=y_{[A]} & e_{[B]}=y_{[B]}-\hat y\quad\#\ \hat y\text{ is label}\\
\text{compute\_gradient} & g_{w_A,[A]},\ g_{w_B,[A]} & g_{w_A,[B]},\ g_{w_B,[B]}\\
& w_{A,[A]}=w_{A,[A]}-\mu\cdot g_{w_A,[A]} & w_{A,[B]}=w_{A,[B]}-\mu\cdot g_{w_A,[B]}\quad\#\ \mu\text{ is learning rate}\\
& w_{B,[A]}=w_{B,[A]}-\mu\cdot g_{w_B,[A]} & w_{B,[B]}=w_{B,[B]}-\mu\cdot g_{w_B,[B]}\\
\text{compute\_loss} & | & L
\end{array}
$$

## 2.2. cal_prediction

$$
\begin{array}{lll}
\qquad\qquad\qquad\qquad\qquad & \text{A(Host)}\qquad\qquad\qquad\qquad\qquad\qquad\qquad & \text{B(Guest)}\qquad\qquad\qquad\qquad\qquad\qquad\qquad\\
\text{\_cal\_z\_in\_share} & z_{[A]} & z_{[B]}\\
\\
& \text{SEND }[[z_{[A]}]]^A & |\\
& | & [[z]]^A=[[z_{[A]}]]^A+z_{[B]}\quad\#\text{keep to self.}\\
& | & [[y]]^A=0.5+0.25\cdot[[z]]^A\quad\#\text{ approx sigmoid}\\
& | & [[e]]^A=[[y]]^A-\hat y\quad\#\text{ keep to self.}\\
& | & \text{share }[[y]]^A\rightarrow y_{[B]},\ [[ y_{[A]}]]^A\\
& | & \text{SEND }[[y_{[A]}]]^A\\
& y_{[A]} & | \\
\\
\text{return} & y_{[A]} & y_{[B]}\\
\end{array}
$$

### 2.2.1. _cal_z_in_share

$$
\begin{array}{lll}
\qquad\qquad\qquad\qquad\qquad & \text{A(Host)}\qquad\qquad\qquad\qquad\qquad\qquad\qquad & \text{B(Guest)}\qquad\qquad\qquad\qquad\qquad\qquad\qquad\\
& z_{A,[A]}=x_A\cdot w_{A,[A]} & z_{B,[B]}=x_B\cdot w_{B,[B]}\\
\\
\text{secure\_matrix\_mul\_}& | &\text{SEND }[[w_{A,[B]}]]^B\\
& [[z_{A,[B]}]]^B=x_A\cdot[[w_{A,[B]}]]^B & |\\
& \text{share }[[z_{A,[B]}]]^B\rightarrow z_{A,[B],[A]},\ [[z_{A,[B],[B]}]]^B & |\\
& \text{SEND } [[z_{A,[B],[B]}]]^B& |\\
& | & z_{A,[B],[B]} \\
\text{--------------------}\\
& \text{SEND }[[w_{B,[A]}]]^A & |\\
& | & [[z_{B,[A]}]]^A=x_B\cdot [[w_{B,[A]}]]^A \\
& | & \text{share }[[z_{B,[A]}]]^A\rightarrow z_{B,[A],[B]},[[z_{B,[A],[A]}]]^A\\
& | & \text{SEND }[[z_{B,[A],[A]}]]^A\\
& z_{B,[A],[A]} & | \\
\\
& z_{[A]}=z_{A,[A]}+z_{A,[B],[A]}+z_{B,[A],[A]} & z_{[B]}=z_{B,[B]}+z_{B,[A],[B]}+z_{A,[B],[B]}\\
\text{return} & z_{[A]} & z_{[B]}
\end{array}
$$

## 2.3. compute_gradient

$$
\begin{array}{lll}
\qquad\qquad\qquad\qquad\qquad & \text{A(Host)}\qquad\qquad\qquad\qquad\qquad\qquad\qquad & \text{B(Guest)}\qquad\qquad\qquad\qquad\qquad\qquad\qquad\\
& | & [[g_{w_B}]]^A=[[e]]^A\cdot x_B\\
& | & \text{share }[[g_{w_B}]]^A\rightarrow g_{w_B,[B]},\ [[g_{w_B,[A]}]]^A\\
& | & \text{SEND }[[g_{w_B,[A]}]]^A\\
& g_{w_B,[A]} & |\\
\\
\text{secure\_matrix\_mul\_} & | & \text{SEND }[[e_{[B]}]]^B\\
& [[g_{w_A,[B]}]]^B=[[e_{[B]}]]^B\cdot x_A & |\\ 
& \text{share }[[g_{w_A,[B]}]]^B\rightarrow g_{w_A,[B],[A]},\ [[g_{w_A,[B],[B]}]]^B & | \\
& \text{SEND } [[g_{w_A,[B],[B]}]]^B & |\\
& | & g_{w_A,[B],[B]} \\
\\
& g_{w_A,[A]}=e_{[A]}\cdot x_A+g_{w_B,[B],[A]} & g_{w_A,[B]}=g_{w_A,[B],[B]} \\
\\
\text{return} & g_{w_A,[A]},\ g_{w_B,[A]} & g_{w_A,[B]},\ g_{w_B,[B]}
\end{array}
$$

## 2.4. compute_loss

$$
\begin{array}{lll}
\qquad\qquad\qquad\qquad\qquad & \text{A(Host)}\qquad\qquad\qquad\qquad\qquad\qquad & \text{B(Guest)}\qquad\qquad\qquad\qquad\qquad\qquad\\
& | & \text{share }[[z]]^A\rightarrow z_{[B]},\ [[z_{[A]}]]^A\\
& | & \text{SEND }[[z_{[A]}]]^A\\
& z_{[A]} & | \\
\\
\text{spdz.dot} & \text{(zy)}_{[A]} & \text{(zy)}_{[B]}\\
\\
&|&\text{SEND }[[z_{[B]}]]^B,\ [[z_{[B]}^2]]^B\\
&[[z_{[B]}]]^B,\ [[z_{[B]}^2]]^B & |\\
&[[z]]^B=z_{[A]}+[[z_{[B]}]]^B & | \\
&[[z^2]]^B=z_{[A]}^2+[[z^2_{[B]}]]^B+2z_{[A]}[[z_{[B]}]]^B & |\\
&[[L_{[A]}]]^B=-\ln0.5+0.5[[z]]^B-\text{(zy)}_{[A]}+0.125[[z^2]]^B & |\\
&\text{SEND }[[L_{[A]}]]^{(B)} & | \\
&|&L_{[A]}\\
&|&L=L_{[A]}-\text{(zy)}_{[B]}\\
\\
\text{return} & | & L
\end{array}
$$

### 2.4.1. spdz.dot

```python
# SPDZ
def dot:
    return left.dot(right, target_name)

# FFTensor
def dot(self, other):
    return self.einsum(other, "ij,ik->jk", target_name)
```

FFTensor.einsum
$$
\begin{array}{lll}
\qquad\qquad\qquad\qquad & \text{A(Host)}\qquad\qquad\qquad\qquad\qquad\qquad & \text{B(Guest)}\qquad\qquad\qquad\qquad\qquad\qquad\\
\text{beaver\_triplets} & \text{random sample: } a_A,b_A & \text{random sample}: a_B,b_B\\
& \text{SEND }[[a_A]]^A & \text{SEND }[[a_B]]^B\\
& [[a_B]]^B & [[a_A]]^A\\
& \text{random sample: } r_A & \text{random sample: } r_B\\
& [[a_Bb_A+r_A]]^B=[[a_B]]^B\cdot b_A+r_A & [[a_Ab_B+r_B]]^A=[[a_A]]^A\cdot b_B+r_B\\
& \text{SEND }[[a_Bb_A+r_A]]^B & \text{SEND }[[a_Ab_B+r_B]]^A\\
& a_Ab_B+r_B & a_Bb_A+r_A\\
& c_A=a_Ab_B+r_B+a_Ab_A-r_A & c_B=a_Bb_A+r_A+a_Bb_B-r_B\\
\text{return} & a_{[A]}=a_A,\ b_{[A]}=b_A,\ c_{[A]}=c_A & a_{[B]}=a_B,\ b_{[B]}=b_B,\ c_{[B]}=c_B\\
\\
\text{add then reconstruct} & x+a,\ y+b & x+a,\ y+b\\
\\
& \text{(xy)}_{[A]}=c_{[A]}-a_{[A]}(y+b)-b_{[A]}(x+a)+(x+a)(y+b) & \text{(xy)}_{[B]}=c_{[B]}-a_{[B]}(y+b)-b_{[B]}(x+a)\\
\\
\text{return} & \text{(xy)}_{[A]} & \text{(xy)}_{[B]}
\end{array}
$$
分析：
$$
\begin{align}
\text{(xy)}_{[A]}+\text{(xy)}_{[B]}&=c_{[A]}+c_{[B]}+(x+a)(y+b)-a_{[A]}(y+b)-b_{[A]}(x+a)-a_{[B]}(y+b)-b_{[B]}(x+a)\\
&=c+(x+a)(y+b)-a(y+b)-b(x+a)\\
&=xy
\end{align}
$$

# 3. Some functions

#### share_matrix

```python
def share_matrix(self, matrix_tensor, suffix=tuple()):
    '''
    1. split into two fraction,
    2. keep one, send the other.
    '''
```

#### received_share_matrix

```python
def received_share_matrix(self, cipher, q_field, encoder, suffix=tuple()):
    '''
    Receive, decrypt, return
    '''
```

#### secure_matrix_mul_active

```python
def secure_matrix_mul_active(self, matrix, cipher, suffix=tuple()):
    '''
    Encrypt then remote.
    '''
```

#### secure_matrix_mul_passive

```python
def secure_matrix_mul_passive(self, matrix, suffix=tuple()):
    '''
    1. Receive an encrypted matrix from peer party, 
    2. multiply it by the argument matrix, 
    3. then share the result into two fractions, remote one to peer party, and return the other.
    '''
```

#### secure_matrix_mul

```python
def secure_matrix_mul(self, matrix, cipher=None, suffix=tuple()):
    '''
    - If cipher is passed in:
    	1. encrypt then remote the passed in matrix,
    	2. receive an encrypted matrix from peer, decrypt it and return the result. 
    - else:
    	1. receive an encrypted matrix from peer, 
    	2. multiply it by the passed in matrix,
    	3. share the result
    '''
```



