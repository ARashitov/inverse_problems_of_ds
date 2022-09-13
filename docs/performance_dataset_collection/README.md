# Performance dataset collection 🎯

Performance dataset used to estimate performance of clustering algorithm and train fragmentation models. It plays critical role for our systems and this documentation is needed to explain these methods depending of circumstances

**[CHECKOUT LIMITATIONS REQUIRED TO BE MET BEFORE COLLECTION OF DATASET](./limitations/)**

## Types of collection 🔍

----

There 2 types of performance dataset collection we can make:

1. **Manual** ✍🏼
2. **Semi-automatic** ⚙️

### Manual ✍🏼

Manual performance dataset collection by visualizing GPS records on a kepler map and extraction potential cluster candidates by looking on how much time vehicle spent on some location then verification this location on google map to filter false positives.

#### Below is picture describing essence of manual collection

![rick_dalton](../../diagrams/rick.jpg)

### Semi-automatic ⚙️

Semi-automatic is collection by getting customer plan and preprocessing it to the state, that it can be used for training the models

[Checkout semi-automatic documentation detials](./semi_automatic/)

## Pros & Cons ➕ ➖

----

**Performance**: Manual can exceed and undershoot semi-automatic, if customer provides poor quality data, manual collection is preferred

**Complexity**: Both types have disadvantages, where manual collection is irritating and dull. Where Semi-automatic is complex due to algorithms and steps applied to link source data properly

**Time**: If linking source can become hard it may be easier to perform collection manually
