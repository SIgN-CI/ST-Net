# patient=BC50027

echo "Enter patient number (in BCxxxxx format):"
read patient

echo "Zipping figures for patient ${patient}..."
zip -r ${patient}_figures.zip fig/visualize/${patient}_*.pdf