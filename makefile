download-files:
	@echo "Downloading data files for notebooks"
	@echo "Downloading disaster_declaration_summary.parquet"
	curl -L -o multilabelClassificationModels/disaster_declaration_summary.parquet \
		'https://drive.google.com/uc?export=download&id=1c7upjFyH9g_NbK_-TvhZ3A3HVJzJc0Wz'
	@echo "Download complete!"
	@echo "Downloading DisasterDeclarationsSummaries.csv"
	curl -L -o topicClustering/DisasterDeclarationsSummary.csv \
		'https://drive.google.com/uc?export=download&id=1P3HfhVkEN4g9fZ7LTlKtJTXPgHBMY4qJ'
	@echo "Download complete!"
	@echo "Downloading mission_assignments.parquet"
	curl -L -o multilabelClassificationModels/mission_assignments.parquet \
		'https://drive.google.com/uc?export=download&id=1eWyDfGK5imYsfdV7t7n66G210UVGeEz5'
	@echo "Download complete!"
	@echo "Downloading MissionAssignments.csv"
	curl -L 'https://drive.usercontent.google.com/download?id=11TwVSibdobPkmgH0Yw_QpZBns9djUBr2&confirm=t' -o topicClustering/MissionAssignments.csv
	@echo "Download complete!"
	@echo "Copying MissionAssignments.csv to NLP directory"
	cp topicClustering/MissionAssignments.csv naturalLanguageProcessing
	@echo "Copy complete!"
	@echo "Downloading llm_MA_merge.csv"
	curl -L 'https://drive.usercontent.google.com/download?id=1YWnhbidLPPXKlvFSasI2e90Dpe_PFLld&confirm=t' -o topicClustering/llm_MA_merge.csv
	@echo "Download complete!"
	@echo "Downloading actions_topics_bedrock_full_sow.csv"
	curl -L 'https://drive.usercontent.google.com/download?id=1jyEiPCIKCuMu6xE3qC6k037eSPWesKY1&confirm=t' -o topicClustering/actions_topics_bedrock_full_sow.csv
	@echo "Download complete!"
	@echo "All downloads complete!"
