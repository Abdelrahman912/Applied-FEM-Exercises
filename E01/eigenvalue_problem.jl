### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 4c4985ec-7cd0-4531-86f5-7642293977e9
begin
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ 6319214c-8114-4a16-9b0e-756fac8076d8
using LinearAlgebra

# ╔═╡ 5b363c99-be33-47ce-8c64-f273334df015
begin
	H = 0.7
	B1 = 1 
	B2 = 0.1 
	EA = 2.0e7
	ρA = 1.0
	M = [23.98 0;0.0 0.407]
	K = [6.76e7 -7.7e6; -7.7e6 5.39e6]
end

# ╔═╡ 2ed35c6f-b660-4b76-9568-660df27ce3c7
md"""#### The generatized Eigenvalue problem: 

$\begin{gather}
\boldsymbol{K} \cdot \boldsymbol{\phi} = \lambda \boldsymbol{M} \cdot\boldsymbol{\phi}
\end{gather}$"""

# ╔═╡ 223f35ea-e6f3-4d8b-a42f-cac9caddea09
md"""#### Inverse Iteration
##### Step 1:
Make an intial guess for $\boldsymbol{x_1}$ and compute:

$\begin{gather}
\boldsymbol{y_1} = \boldsymbol{M} \cdot \boldsymbol{x_1}
\end{gather}$""" 

# ╔═╡ eddf0e3f-f2f3-4282-836a-414de5eb0e61
x_1 = [1  ; 0]

# ╔═╡ 460823ac-3b69-4c9a-bc57-0aef3acfebb7
md"""##### Step 2:
Loop starting from n = 1 until convergence:

###### 2.1. Calculate $\bar{\boldsymbol{x}}_{n+1}$:
$\begin{gather}
\boldsymbol{K} \cdot \bar{\boldsymbol{x}}_{n+1} = \boldsymbol{y_n} \Rightarrow \bar{\boldsymbol{x}}_{n+1} = \boldsymbol{K}^{-1} \cdot \boldsymbol{y}_n
\end{gather}$

###### 2.2. Calculate $\bar{\boldsymbol{y}}_{n+1}$:
$\begin{gather}
\bar{\boldsymbol{y}}_{n+1} = \boldsymbol{M} \cdot \bar{\boldsymbol{x}}_{n+1}
\end{gather}$

###### 2.3. Calculate Rayleigh - Quotient $\rho(\bar{\boldsymbol{x}}_{n+1})$:
$\begin{gather}
\rho(\bar{\boldsymbol{x}}_{n+1}) = \frac{\bar{\boldsymbol{x}}_{n+1} \cdot \boldsymbol{y}_n}{\bar{\boldsymbol{x}}_{n+1} \cdot \bar{\boldsymbol{y}}_{n+1}}
\end{gather}$


###### 2.4. Calculate $\boldsymbol{y}_{n+1}$:
$\begin{gather}
\boldsymbol{y}_{n+1} = \bar{\boldsymbol{x}}_{n+1} \frac{1}{\bar{\boldsymbol{x}}_{n+1} \cdot \bar{\boldsymbol{y}}_{n+1}}
\end{gather}$
""" 

# ╔═╡ 488ac317-2050-4b2f-9618-ac7d95cd6336
function inverse_iteration_loop(M,K,x_1)
	notconverged = true
	rho_old = 0.0
	rho_new = 0.0
	y = M * x_1
    TOL = 1.0e-6

	while(notconverged)
		xbar = inv(K) *  y
		ybar = M * xbar
		rho_old = rho_new
		rho_new = (xbar' * y)/(xbar' * ybar)
		y = ybar * (1/(xbar' * ybar))
		err = abs(rho_new - rho_old)/rho_new 
		if(err ≤ TOL)
			notconverged = false
		end
	end
	rho_new, y	
end

# ╔═╡ d27b3f46-a149-4078-9a43-7b24812c12e0
md"""##### Step 3: Results:

###### 3.1. Calculate $\lambda_1$:
$\begin{gather}
\lambda_1 \leftarrow \rho
\end{gather}$


###### 3.2. Calculate $\boldsymbol{\phi}_k$:
$\begin{gather}
\boldsymbol{y}_{n+1} = \boldsymbol{M} \cdot \boldsymbol{\phi}_k \Rightarrow \boldsymbol{\phi}_k = \boldsymbol{M}^{-1} * \boldsymbol{y}_{n+1}
\end{gather}$


###### 3.3. Normalize $\boldsymbol{\phi}_k$:

$\begin{gather}
\boldsymbol{\phi}_k \leftarrow \frac{\boldsymbol{\phi}_k}{|\boldsymbol{\phi}_k|}
\end{gather}$"""

# ╔═╡ bb6ab6b4-085f-4585-be54-5bb7caad7073
function inverse_iteration(M,K,x_1)
	λ , y = inverse_iteration_loop(M,K,x_1)
	ϕ = inv(M) * y
	ϕ = ϕ / norm(ϕ)
	return λ , ϕ
end

# ╔═╡ fa96fb3c-c2a4-4507-867c-d54c1ccb3e92
inverse_iteration(M,K,x_1)

# ╔═╡ Cell order:
# ╠═4c4985ec-7cd0-4531-86f5-7642293977e9
# ╠═6319214c-8114-4a16-9b0e-756fac8076d8
# ╠═5b363c99-be33-47ce-8c64-f273334df015
# ╟─2ed35c6f-b660-4b76-9568-660df27ce3c7
# ╟─223f35ea-e6f3-4d8b-a42f-cac9caddea09
# ╠═eddf0e3f-f2f3-4282-836a-414de5eb0e61
# ╟─460823ac-3b69-4c9a-bc57-0aef3acfebb7
# ╠═488ac317-2050-4b2f-9618-ac7d95cd6336
# ╟─d27b3f46-a149-4078-9a43-7b24812c12e0
# ╠═bb6ab6b4-085f-4585-be54-5bb7caad7073
# ╠═fa96fb3c-c2a4-4507-867c-d54c1ccb3e92
