download-files:
	@echo "Downloading data files for notebooks"
	@echo "Downloading disaster_declaration_summary.parquet"
	curl -L -o disaster_declaration_summary.parquet \
		'https://drive.google.com/uc?export=download&id=1c7upjFyH9g_NbK_-TvhZ3A3HVJzJc0Wz'
	@echo "Download complete!"
	@echo "Downloading DisasterDeclarationsSummary.csv"
	curl -L -o DisasterDeclarationsSummary.csv \
		'https://drive.google.com/uc?export=download&id=1P3HfhVkEN4g9fZ7LTlKtJTXPgHBMY4qJ'
	@echo "Download complete!"
	@echo "Downloading mission_assignments.parquet"
	curl -L -o mission_assignments.parquet \
		'https://drive.google.com/uc?export=download&id=1eWyDfGK5imYsfdV7t7n66G210UVGeEz5'
	@echo "Download complete!"
	@echo "Downloading MissionAssignments.csv"
	curl -L 'https://drive.usercontent.google.com/download?id=11TwVSibdobPkmgH0Yw_QpZBns9djUBr2&confirm=t' -o MissionAssignments.csv
	@echo "All downloads complete!"
