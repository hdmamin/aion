.PHONY nanochat

# Pull changes from karpathy's original nanochat repo.
nanochat:
	git fetch nanochat
	git subtree pull --prefix=nanochat nanochat master --squash

nb:
	cp -i notebooks/TEMPLATE.ipynb notebooks/todo_untitled.ipynb
