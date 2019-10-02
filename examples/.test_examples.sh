for i in examples/*.ipynb
do
  echo "Running Notebook: $i"
  jupyter nbconvert --stdout --execute $i > /dev/null
  echo "Success"
  echo ""
done

for i in examples/*.py
do
  echo "Running Script: $i"
  python $i > /dev/null
  echo "Success"
  echo ""
done
