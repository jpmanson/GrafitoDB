WITH "examples/datasets/belgian_beers.xml" AS path
CALL apoc.load.xml(path, ".//beer") YIELD value
WITH value.brand._text AS BeerBrand,
     value.type._text AS BeerType,
     value.alcohol._text AS AlcoholPercentage,
     value.brewery._text AS Brewery,
     value.timeframe._text AS Timeframe
MERGE (bt:BeerType {name: coalesce(BeerType, "Unknown")})
MERGE (bb:BeerBrand {name: coalesce(BeerBrand, "Unknown")})
SET bb.Timeframe = coalesce(Timeframe, "Unknown")
MERGE (br:Brewery {name: coalesce(Brewery, "Unknown")})
MERGE (ap:AlcoholPercentage {value: coalesce(AlcoholPercentage, "Unknown")})
MERGE (bb)-[:HAS_ALCOHOLPERCENTAGE]->(ap)
MERGE (bb)-[:IS_A]->(bt)
MERGE (bb)<-[:BREWS]-(br);
