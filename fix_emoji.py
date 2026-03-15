from huggingface_hub import HfApi

api = HfApi()

with open("src/dashboard/app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Remove BOM if present
content = content.replace("\ufeff", "")

# Fix sidebar radio options
content = content.replace('"🏠  Overview"',          '"Overview"')
content = content.replace('"📊  Model Performance"', '"Model Performance"')
content = content.replace('"🔍  Feature Importance"','"Feature Importance"')
content = content.replace('"💹  Backtest Results"',  '"Backtest Results"')
content = content.replace('"🔎  Transcript Explorer"','"Transcript Explorer"')

# Fix page routing
content = content.replace('if page == "🏠  Overview":',           'if page == "Overview":')
content = content.replace('elif page == "📊  Model Performance":', 'elif page == "Model Performance":')
content = content.replace('elif page == "🔍  Feature Importance":','elif page == "Feature Importance":')
content = content.replace('elif page == "💹  Backtest Results":',  'elif page == "Backtest Results":')
content = content.replace('elif page == "🔎  Transcript Explorer":','elif page == "Transcript Explorer":')

api.upload_file(
    path_or_fileobj=content.encode("utf-8"),
    path_in_repo="app.py",
    repo_id="Rajveer234/finsight",
    repo_type="space",
)
print("Done. Refresh Space in 30 seconds.")