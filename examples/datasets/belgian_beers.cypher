WITH "https://nl.wikipedia.org/wiki/Lijst_van_Belgische_bieren" as url
    CALL apoc.load.html(url, {
        brand: "table.wikitable tbody tr td:eq(0)",
        beertype: "table.wikitable tbody tr td:eq(1)",
        alcoholpercentage: "table.wikitable tbody tr td:eq(2)",
        brewery: "table.wikitable tbody tr td:eq(3)",
        timeframe: "table.wikitable tbody tr td:eq(4)"
        }) yield value
WITH value, size(value.brand) as rangeup
UNWIND range(0,rangeup) as i
WITH value.brand[i].text as BeerBrand, value.brewery[i].text as Brewery,
     value.alcoholpercentage[i].text as AlcoholPercentage,
     value.beertype[i].text as BeerType, value.timeframe[i].text as Timeframe
MERGE (bt:BeerType {name: coalesce(BeerType,"Unknown")})
MERGE (bb:BeerBrand {name: coalesce(BeerBrand,"Unknown")})
SET bb.Timeframe = coalesce(Timeframe,"Unknown")
MERGE (br:Brewery {name: coalesce(Brewery,"Unknown")})
MERGE (ap:AlcoholPercentage {value: coalesce(AlcoholPercentage,"Unknown")})
MERGE (bb)-[:HAS_ALCOHOLPERCENTAGE]->(ap)
MERGE (bb)-[:IS_A]->(bt)
MERGE (bb)<-[:BREWS]-(br);
