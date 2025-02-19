import numpy as np
import pandas as pd
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--prev_input", type=str, default='')
	parser.add_argument("--inputs", nargs='+', type=str, default=['input1.csv','input2.csv'])
	parser.add_argument("--output_fig", type=str, default='output.png')
	parser.add_argument("--output", type=str, default='output.pkl')
	parser.add_argument("--reward_name", type=str, default='f_osc')
	parser.add_argument("--output_max_reward", type=str, default='output_max_reward')
	parser.add_argument("--output_min_reward", type=str, default='output_min_reward')
	params = parser.parse_args()

	if params.prev_input != '':
		with open(params.prev_input, 'rb') as f:
			data = pickle.load(f)
	else:
		data = []
	for i in range(len(params.inputs)):
		data += [pd.read_csv(params.inputs[i])]
		data[-1]['N'] = data[-1]['N'].astype(int)
		data[-1]['loss'] = data[-1]['loss'].astype(float)
		data[-1][params.reward_name] = data[-1][params.reward_name].astype(float)
		data[-1]['diversity-10'] = data[-1]['diversity-10'].astype(float)
		data[-1]['closest_smiles_sim'] = data[-1]['closest_smiles_sim'].astype(float)

	t = np.zeros(len(data))
	max_top1_reward = np.zeros(len(data))
	top1_reward = np.zeros(len(data))
	top10_reward = np.zeros(len(data))
	top100_reward = np.zeros(len(data))
	top1_sim = np.zeros(len(data))
	top10_sim = np.zeros(len(data))
	top100_sim = np.zeros(len(data))
	top10_div = np.zeros(len(data))
	for i in range(len(data)):
		if i == 0:
			t[i] = data[i]['N'].iloc[0]
		else:
			t[i] = t[i-1] + data[i]['N'].iloc[0]
		if i==0 or data[i][params.reward_name].max() > max_top1_reward[i-1]:
			max_top1_reward[i] = data[i][params.reward_name].max()
		else:
			max_top1_reward[i] = max_top1_reward[i-1]
		top1_reward[i] = data[i][params.reward_name].max()
		top10_reward[i] = data[i].nlargest(10, params.reward_name)[params.reward_name].mean()
		top100_reward[i] = data[i].nlargest(100, params.reward_name)[params.reward_name].mean()
		top1_sim[i] = data[i].nlargest(1, params.reward_name)['closest_smiles_sim'].iloc[-1]
		top10_sim[i] = data[i].nlargest(10, params.reward_name)['closest_smiles_sim'].mean()
		top100_sim[i] = data[i].nlargest(100, params.reward_name)['closest_smiles_sim'].mean()
		top10_div[i] = data[i]['diversity-10'].iloc[0]

	cmap = plt.cm.get_cmap('Dark2')
	colors = [cmap(i) for i in np.linspace(0, 1, 4)]

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), layout='constrained')
	ax.set_ylabel("Oscillator strength")
	ax.set_xlabel("Number of samples evaluated")
	#ax.plot(t, max_top1_reward, color ="black", label="Max Reward", zorder=4) 
	ax.plot(t, top1_reward, color=colors[0], label="Top-1", zorder=3)
	ax.plot(t, top10_reward, color=colors[1], label="Top-10", zorder=2)
	ax.plot(t, top100_reward, color=colors[2], label="Top-100", zorder=1)
	ax.legend()
	fig.savefig('output_reward.png', dpi=450)

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), layout='constrained')
	ax.set_ylabel("Top-10 Diversity")
	ax.set_xlabel("Number of samples evaluated")
	ax.plot(t, 1-top10_div, color = colors[1]) 
	fig.savefig('output_diversity.png', dpi=450)
	
	# Save outputs
	with open(params.output, 'wb') as f:
		pickle.dump(data, f)
	with open(params.output_max_reward, 'w') as f:
		f.write(str(max(top1_reward)))
	with open(params.output_min_reward, 'w') as f:
		f.write(str(max(top100_reward)))