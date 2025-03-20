### num_agents, maps

``` 
batch['num_agents'] 是一个batch中不同场景的agents数量;
batch['maps']则是(B,N,3,224,224),其中N是batch['num_agents']的最大值,如果场景中没有那么多agent,那么可视化的地图就是全黑色的
 ```
 ### 可视化地图
 ```bash
agent_idx =0 
map = batch['maps'][batch_idx, agent_idx].permute(1, 2, 0).cpu().numpy()
custom_map = np.ones((map.shape[0], map.shape[1], 3))
custom_map[map[:,:,2] > 0] = np.array([200, 211, 213]) / 255.
custom_map[map[:,:,1] > 0] = np.array([164, 184, 196]) / 255.
custom_map[map[:,:,0] > 0] = np.array([164, 184, 196]) / 255.
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(custom_map)
 ```

### raster
```
batch['raster_from_world']=batch['rasters_from_world_tf']
```



### 历史轨迹
```
batch['agent_hist'] (B,N,T_hist,8)
batch['agent_hist_extent'] (B,N,T_hist,3)
```
