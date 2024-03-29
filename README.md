# RecSys: SKU Collaborative filtering 

1. User-Item Matrix
2. Make Item Embeddings
3. ItemUniquiness & FastAPiService

## UserItemMatrix

UserItemMatrix class that creates a User-Item matrix for aggregated data.

The class contains 5 methods - they return information about the object. They are immutable properties of the class.

Since they are @property - calculations need to be done in the init method, returning private variables within each method.

The user_map and item_map methods return mappings with sorted keys - the original user_id and item_id.


There are 3 main ways to build embeddings for items:

1. Based on content (picture, title, item description).
2. Based on sales history (history of who bought what items in the past).
3. Based on user sessions (history of what products our users are looking at).

We will focus on the second method.

Its essence is as follows:

Each product is characterized by the users who bought it.
Each user is characterized by the products they bought.
There seems to be a two-way relationship between goods and users. Methods based on the premise "similar users - buy similar products" are called collaborative filtering (co-filtering).

So, if we want to understand how similar two users are, we can count how many of each product each user bought - and count the overlap. On the other hand, if we want to understand how similar two products are, we can just compare all users who bought them.

Both problems can be solved at once by representing the original sales history as a User-Item matrix, where element x ij is the number of purchases by user i of product j. Then, by calculating the cosine proximity between the two rows we will know how similar two users are; and between the columns we will know how similar two products are.

![User-Item matrix Karpov Courses style](https://github.com/apovalov/SKU_Pricing-Recommendations/assets/43651275/28f3ec5b-cbfc-4c6c-a6ce-f340447d34cb)


## Normalization

What if the average sales for a few User-Item pairs are 10 to 100 times larger for all others?

As on any other tabular data, we can perform a normalization of the User-Item matrix.

1. Line by line.
2. Column by column.
3. Based on TF-IDF
4.Based on BM-25. You need to use a modified version of TF-IDF, which is used in BM-25.

TF-IDF (from TF - term frequency, IDF - inverse document frequency) is a statistical index used to evaluate the importance of a word in the context of a document that is part of a corpus (document collection).

TF is the ratio of the number of occurrences of a word to the total number of words in the document.

IDF - inverse frequency, is the ratio of the total number of documents in the collection to the number of documents containing the word. The logarithm reduces the weight of commonly used wordsWhat if the average sales for a few User-Item pairs are 10 to 100 times greater for all others?

As on any other tabular data, we can perform a normalization of the User-Item matrix.

Line by line.
Column by column.
Based on TF-IDF, apply the Bag of Word analogy described above.
Based on BM-25. You need to use a modified version of TF-IDF, which is used in BM-25.

TF-IDF (from TF - term frequency, IDF - inverse document frequency) is a statistical index used to evaluate the importance of a word in the context of a document that is part of a corpus (document collection).

TF is the ratio of the number of occurrences of a word to the total number of words in the document.

IDF - inverse frequency, is the ratio of the total number of documents in the collection to the number of documents containing the word. Logarithm reduces the weight of commonly used words

![1234](https://github.com/apovalov/SKU_Pricing-Recommendations/assets/43651275/79e3c5e7-d9fd-46cc-afd7-83d9919fdb5e)


Rows - documents, columns - words. The intersection is the number of occurrences of words.
If we imagine that users are documents and words are goods, then we can apply the formula for calculating TF-IDF coefficients for each user-good pair.

To calculate the distance between two goods (e.g., Euclidean or cosine distance), we need to learn how to map them into a multidimensional space ("goods space"). A point in this space (essentially a vector of numbers), is called an embedding.

![Screenshot 2024-02-04 at 21 38 08](https://github.com/apovalov/RecSys_SKU_Embs_TF_IDF/assets/43651275/c8a62d7b-21dd-4c9e-b361-e4cc6b1340f4)
![Screenshot 2024-02-04 at 21 38 28](https://github.com/apovalov/RecSys_SKU_Embs_TF_IDF/assets/43651275/1a918d0d-0221-4e8d-a96a-91ab4c8ab15e)

If we think of embedding as a directional vector, then cosine similarity shows how much the two vectors are co-directional (given possibly different vector lengths). A popular metric for word embeddings.

![Screenshot 2024-02-04 at 21 39 10](https://github.com/apovalov/RecSys_SKU_Embs_TF_IDF/assets/43651275/d78fd832-c9af-4cfa-9ab4-6071efbd496a)


## SimilarItems

Similar Items Price
When we estimate the number of sales in dynamic pricing or demand forecasting tasks, price and related attributes usually provide the strongest signal:

The price of the item in the past.
The price of similar items.
A competitor's price.
The price of a discounted item.
Purchase price.

And also derivatives of prices, e.g:

Discount = price / price_before_discount - 1.
Margin = 100% * (1 - purchase_price / price).
Relation_to_market = price / competitor_price.

similarity - counts pairwise similarities between all embeddings, returning a dictionary of similarities.
knn - the function takes as input the result of the similarity function and the top parameter - the number of nearest neighbors. It outputs a dictionary with item_id - list of top nearest items pairs.
knn_price - the function takes as input the result of the knn function and the price dictionary with prices for each item. The output is the weighted average price of the top nearest neighbors.
transform - transforms the original embedding dictionary into a dictionary with new prices for all products.



## Item uniqueness

knn_uniqueness - evaluates the uniqueness of each product (uniqueness), how far the product's embedding is removed (isolated) from other embeddings. The more distant an embedding is in n-dimensional space from other embeddings, the greater its uniqueness metric value.

kde_uniqueness - Kernel Density Estimation

We "smear" the distribution around each object by representing it with a Gaussian (multivariate normal distribution, essentially a sphere) with a given variance (a given radius). This is called a Kernel-trick.
We then add up all the distributions that have been created.
Those places where many Gaussians (many embeddings) have been lumped together will get an overlap of several distributions. After normalization (the total density of the distribution is 1), we get the density estimate we are looking for.

For us, the density at the point where the embedding is located will be an estimate of its uniqueness.

group_diversity - metric for calculating the diversity of a group of products


## FastAPI service
![Новый проект (8)](https://github.com/apovalov/SKU_Pricing-Recommendations/assets/43651275/719ee629-936e-4983-af82-b85780bf87c8)

load_embeddings - loads all embeddings every 10 seconds.

GET request /uniqueness - which asks to evaluate a selection of items. Expects item ID → uniqueness to be displayed in the response.

GET request /diversity - which accepts a selection of items as input. It returns a diversity metric and a verdict to "reject" or "accept" this output.


1. Matrix: User - Item
2. Matrix Normalization // Rows/Columns or TF-IDF or BM-25
3. Matrix Factorization -> Embeddings
