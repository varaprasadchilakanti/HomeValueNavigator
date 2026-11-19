# Third-party notices

These are not covered by this repository's MIT license.

## Dataset

`data/bengaluru/listings.csv` is byte-identical to `bengaluru_house_prices.csv` in the
codebasics course's [code repository](https://github.com/codebasics/py/tree/master/DataScience/BangloreHomePrices),
for the course [Real Estate Price Prediction](https://codebasics.io/courses/data-science-project-real-estate-price-prediction);
its SHA-256 is recorded as `data_sha256` in `model/bengaluru/model.json`. That repository's
readme says the data is from kaggle.com and states no license. A Kaggle dataset,
[Bengaluru House price data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data),
with the same columns and row count, is listed under CC0 1.0 (public domain dedication);
that the two are the same data is an inference, not verified. The dedication is the
uploader's statement; who collected the listings, and when, is not recorded.

If you hold rights in these listings and object to their redistribution, open an issue and
the file will be removed.

## Method

The cleaning rules and the project's starting shape follow the course. The training
script, evaluation, model format, API, page and tests in this repository are its own. The
fitted models in `model/` are derived from the dataset above; they are offered under this
repository's license to the extent the dataset's terms allow, which they do if it is the
CC0 1.0 dataset listed on Kaggle.

codebasics, Dhaval Patel and Kaggle are named for attribution only; none of them has
reviewed or endorses this project.
