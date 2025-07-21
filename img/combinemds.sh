# Clear or create a fresh output file
> combined_drift_report.md

# Loop over sorted files by numeric epochs
for epoch in 0 1 5 10 25 50 100 200; do
    for term in head tail relation; do
        file="drift_data_${epoch}epochs_${term}.md"
        if [ -f "$file" ]; then
            echo "Adding $file to report..."

            # Add a clean heading
            echo -e "\n\n## Drift Data after ${epoch} Epochs (${term^})\n\n" >> combined_drift_report.md

            # Append the file content
            cat "$file" >> combined_drift_report.md

            # Optional: Add a horizontal separator
            echo -e "\n\n---\n\n" >> combined_drift_report.md
        else
            echo "Warning: $file not found, skipping."
        fi
    done
done
