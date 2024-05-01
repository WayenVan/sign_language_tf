# csi_sign_language

$$
Score(k)=Softmax_k \frac{\lVert p_{i,j,c,k} \rVert_2^{i,j,c}}{\sqrt{I*J*C}} \\
\hat{p}_{i, j, c} = \sum_k Score(k) * p_{i,j,c,k}
$$

