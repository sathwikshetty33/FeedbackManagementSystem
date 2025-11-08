from fastmcp import FastMCP
import pandas as pd
from io import BytesIO
import requests

mcp = FastMCP("feature-analyzer")

@mcp.tool()
def analyze_features(google_sheet_url: str, features: list[str]):
    """
    Takes a Google Sheet URL and list of feature names.
    Loads the data using pandas, determines data types,
    and returns analysis for each feature.
    """
    try:
        # Convert Google Sheet URL to CSV export link
        if "edit" in google_sheet_url:
            csv_url = google_sheet_url.replace("/edit", "/export?format=csv")
        elif "spreadsheets" in google_sheet_url:
            csv_url = google_sheet_url + "/export?format=csv"
        else:
            csv_url = google_sheet_url  # assume direct CSV link

        # Load the data
        response = requests.get(csv_url)
        response.raise_for_status()
        df = pd.read_csv(BytesIO(response.content))

        results = {}
        for col in features:
            if col not in df.columns:
                results[col] = {"error": "Column not found"}
                continue

            dtype = str(df[col].dtype)
            analysis = {"type": dtype}

            if pd.api.types.is_numeric_dtype(df[col]):
                analysis.update({
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std_dev": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                })
            else:
                analysis.update({
                    "mode": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    "unique_count": int(df[col].nunique()),
                    "most_frequent": df[col].value_counts().idxmax() if not df[col].empty else None,
                })

            results[col] = analysis

        # Include column types overview
        results["columns"] = {
            col: str(df[col].dtype) for col in df.columns
        }

        return results

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run()
