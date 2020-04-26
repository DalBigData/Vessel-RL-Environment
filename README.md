# Vessel Environment

## How to run server?
```bash
cd vessel_server
edit setting.py
edit config_vessels.py
python3 main.py
```

## How to run vessel agent that use table?
```bash
cd vessel_agents
python3 simple.py
```
The agent's model is table, it saves the table in setting.result_path.

## How to run vessel agent that use deep q?
```bash
cd vessel_agents
python3 deep_q_agent.py
```
The agent's model is NN, and model is trained by target network and buffer.
this agent saves it's model weight in setting.result_path.

## How to show environment?
```bash
cd vessel_show
python3 show.py path file [pic, gif, show]
```
* pic: show picture and save it

* gif: save gif file

* show: show gif


