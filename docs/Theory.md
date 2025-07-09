# Embedding Methodologies



## General graph Theory

> [!NOTE] Definition GRAPH
> Let $V$ be a countable set.   
> Then $(V, E)$ with $E\subset V\times V$ is a **_Graph_** 

note, that graphs defined as such are not directed, i.e. the for two nodes $(v_1, v_2) \in V^2$ we have $e_1 = (v_1,v_2) = (v_2,v_1) = e_2$, which leads to the next definition:

> [!NOTE] Definition DIRECTED GRAPH
> Let $(V, E)$ be a Graph as defined above.  
> The Graph is a directed **_directed graph_**, if $(v_1,v_2)\neq (v_2,v_1)$


> [!NOTE] Definition 
> 
>Let $\mathcal{G}_1= (V_1, E_1)$ and $\mathcal{G}_2 = (V_2,E_2)$ be 2 directed graphs and $\vert V_1\vert$ = n, then 
>  - the adjacency matrix $A_{G_1}$ is defined as the Matric $A_{G_1}\in\mathbb{R^{n\times n}}$ with entries $a_{i,j} = \#\lbrace e\in E: e = (v_i,v_j)\rbrace$ for $v_i, v_j\in V$
>  - the two graphs are isomorphic, if $\exist \varphi:V_1\rightarrow V_2$ bijective, such that $e=(v_1,v_2)\in E_1\Rightarrow (\varphi(v_1),\varphi(v_2))\in E_2$
>  - For two isomorphic Graphs $G_1$ and $G_2$ with adjacency matrices $A_1$ and $A_2$ there exists a permutation matrix $P$, such that $P(A_1) = A_2$


## Application to Ontology Embeddings and Ontology Alignment

### Ontology Embedding

### Ontology Alignment

We wll be looking into a simplified version of the Ontology first, by only looking at Classes (i.e. Hierarchy and disjointness) and the Objectproperties.
Let $O_1$ and $O_2$ be Ontologies with Classes $C_1, C_2$, Objectproperties $P_1,P_2$ and $\vert C_1\vert = n,\ \vert C_2 \vert = m,\ n\leq m$.
Creating an Ontology Alignment mathematically boils down to finding isomorphic subgraphs, which is generally an open Problem in mathematics.
To Identify a potential bijection between the classes, we embed the labels and definitions, calculate the cosine distance for both and then calculate the reciprocal rank fusion. Then, each Class and Property of the smaller Ontology "points" to the corresponding Ressource with the highest value in the other Ontology. 
This gives us a permutation matrix $P = P_{C_1, C_2}$ depending on the set of classes in each ontology and their previously computed cosine distances. The Goal is now to find 

```math
\min_{C_1,C_2}\Vert A_1- P(A_2)\Vert
```

