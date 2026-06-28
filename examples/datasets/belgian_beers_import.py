from pathlib import Path

from grafito import GrafitoDatabase


SCRIPT_PATH = Path(__file__).with_name("belgian_beers.cypher")


def main() -> None:
    db = GrafitoDatabase(":memory:")
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    db.execute_script(script)

    stats_queries = {
        "total_nodes": "MATCH (n) RETURN COUNT(n) AS total_nodes",
        "total_relationships": "MATCH ()-[r]->() RETURN COUNT(r) AS total_relationships",
        "beer_brands": "MATCH (n:BeerBrand) RETURN COUNT(n) AS beer_brands",
        "beer_types": "MATCH (n:BeerType) RETURN COUNT(n) AS beer_types",
        "breweries": "MATCH (n:Brewery) RETURN COUNT(n) AS breweries",
        "alcohol_percentages": "MATCH (n:AlcoholPercentage) RETURN COUNT(n) AS alcohol_percentages",
        "has_alcoholpercentage": "MATCH ()-[r:HAS_ALCOHOLPERCENTAGE]->() RETURN COUNT(r) AS has_alcoholpercentage",
        "is_a": "MATCH ()-[r:IS_A]->() RETURN COUNT(r) AS is_a",
        "brews": "MATCH ()-[r:BREWS]->() RETURN COUNT(r) AS brews",
    }

    for label, query in stats_queries.items():
        result = db.execute(query)
        print(f"{label}: {result[0] if result else {}}")

    db.close()


if __name__ == "__main__":
    main()
