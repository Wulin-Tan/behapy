import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from scipy import stats
import os
import cv2

def TrajectoryMA(data,bodyPart,bodyPartBox = None, **kwargs):
    """
    Plots the trajectory of the determined body part.
    The input file MUST BE the .h5 format!

    Parameters
    ----------
    data : pandas DataFrame
        The input tracking data in h5/hdf format.
    bodyPart : str
        Body part you want to plot the tracking.
    bodyPartBox : str
        The body part you want to use to estimate the limits of the environment, 
        usually the base of the tail is the most suitable for this determination.
    animals: list
        If you have multi-animal data and want to plot only one, just pass in this 
        variable a list with the name of the designated animal (e.g. animals ['rat1']). 
        This will allow you to plot using 'fig,axs' and plot the animals separately.
    start : int, optional
        Moment of the video you want tracking to start, in seconds. If the variable 
        is empty (None), the entire video will be processed.
    end : int, optional
        Moment of the video you want tracking to end, in seconds. If the variable is 
        empty (None), the entire video will be processed.
    fps : int
        The recording frames per second.
    cmapType : str, optional
        matplotlib colormap.
    figureTitle : str, optional
        Figure title.
    hSize : int, optional
        Determine the figure height size (x).
    wSize : int, optional
        Determine the figure width size (y).
    fontsize : int, optional
        Determine of all font sizes.
    invertY : bool, optional
        Determine if de Y axis will be inverted (used for DLC output).
    limit_boundaries : bool, optional.
        Limits the points to the box boundary.
    xLimMin : int, optional
      Determines the minimum size on the axis X.
    xLimMax : int, optional
        Determines the maximum size on the axis X.
    yLimMin : int, optional
        Determines the minimum size on the axis Y.
    yLimMax : int, optional
        Determines the maximum size on the axis Y.
    saveName : str, optional
        Determine the save name of the plot.        
    figformat : str, optional
        Determines the type of file that will be saved. Used as base the ".eps", 
        which may be another supported by matplotlib. 
    res : int, optional
        Determine the resolutions (dpi), default = 80.
    ax : fig, optional
        Creates an 'axs' to be added to a figure created outside the role by the user.
    fig : fig, optional
        Creates an 'fig()' to be added to a figure created outside the role by the user.
    joint_plot bool, optional
        If true it will plot all trajectories in a single plot, ideal for multi-animal 
        tracking of several setups (eg. openfield). If false, will plot each animal 
        separately.

    Returns
    -------
    out : plot
        The output of the function is the figure with the tracking plot of the 
        selected body part.

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat

    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations."""

    saveName= kwargs.get('saveName')
    start= kwargs.get('start')
    end= kwargs.get('end')
    figureTitle = kwargs.get('figureTitle')
    fps = kwargs.get('fps')
    ax = kwargs.get('ax')
    limit_boundaries = kwargs.get('limit_boundaries')
    xLimMin = kwargs.get('xLimMin')
    xLimMax = kwargs.get('xLimMax')
    yLimMin = kwargs.get('yLimMin')
    yLimMax = kwargs.get('yLimMax')
    joint_plot = kwargs.get('joint_plot')
    animals = kwargs.get('animals')
    cmapType = kwargs.get('cmapType')
    hSize = kwargs.get('hSize')
    wSize = kwargs.get('wSize')
    fontsize = kwargs.get('fontsize')
    invertY = kwargs.get('invertY')
    figformat = kwargs.get('figformat')
    res = kwargs.get('res')
    fig = kwargs.get('fig')
    if fig is None and ax is not None:
        fig = ax.figure

    if type(limit_boundaries) == type(None):
      limit_boundaries = False
    if type(fps) == type(None):
      fps = 30
    if type(cmapType) == type(None):
      cmapType = 'viridis'
    if type(hSize) == type(None):
      hSize = 6
    if type(wSize) == type(None):
      wSize = 8
    if type(fontsize) == type(None):
      fontsize = 15
    if type(invertY) == type(None):
      invertY = True
    if type(figformat) == type(None):
      figformat = '.eps'
    if type(res) == type(None):
      res = 80

    if type(animals) == type(None):
        animals = list(set([i[0] for i in list(set(data[data.columns[0][0]].columns))]))

    animals_data = {}

    c = []
    
    for i in range(len(animals)):
      animals_data[animals[i]] = data.xs(animals[i], level="individuals", axis=1)

      values = (animals_data[animals[i]].iloc[2:,1:].values).astype(float)
      lista1 = (animals_data[animals[i]].iloc[0][1:].values +" - " + animals_data[animals[i]].iloc[1][1:].values).tolist()

      if type(start) == type(None):
          x = values[:,lista1.index(bodyPart+" - x")]
          y = values[:,lista1.index(bodyPart+" - y")]
      else:
          init = int(start*fps)
          finish = int(end*fps)
          x = values[:,lista1.index(bodyPart+" - x")][init:finish]
          y = values[:,lista1.index(bodyPart+" - y")][init:finish]
      
      animals_data[animals[i]+"_x"] = x
      animals_data[animals[i]+"_y"] = y
      
      c.append(np.linspace(0, x.size/fps, x.size))

    cmap = plt.get_cmap(cmapType)

    if type(bodyPartBox) == type(None):
      esquerda = xLimMin
      direita = xLimMax
      baixo = yLimMin
      cima = yLimMax
    else:
      #TODO: check this part (bodyPartBox for multi animal)
      esquerda = values[:,lista1.index(bodyPartBox+" - x")].min()
      direita = values[:,lista1.index(bodyPartBox+" - x")].max()
      baixo = values[:,lista1.index(bodyPartBox+" - y")].min()
      cima = values[:,lista1.index(bodyPartBox+" - y")].max()

    if limit_boundaries:
        for j in range(len(animals)):
          testeX = []
          x = animals_data[animals[j]+"_x"]
          for i in range(len(x)):
              if x[i] >= direita:
                  testeX.append(direita)
              elif x[i] <= esquerda:
                  testeX.append(esquerda)
              else:
                  testeX.append(x[i])
          animals_data[animals[j]+"_x"] = testeX

          testeY = []
          y = animals_data[animals[j]+"_y"]
          for i in range(len(y)):
              if y[i] >= cima:
                  testeY.append(cima)
              elif y[i] <= baixo:
                  testeY.append(baixo)
              else:
                  testeY.append(y[i])
          animals_data[animals[j]+"_y"] = testeY
          
    if joint_plot:
      if type(ax) == type(None): 
          plt.figure(figsize=(wSize, hSize), dpi=res)
          plt.title(figureTitle, fontsize=fontsize)
          for i in range(len(animals)):
            plt.scatter(animals_data[animals[i]+"_x"], animals_data[animals[i]+"_y"], c=c[i], cmap=cmap, s=3)
          plt.plot([esquerda,esquerda] , [baixo,cima],"k")
          plt.plot([esquerda,direita]  , [cima,cima],"k")
          plt.plot([direita,direita]   , [cima,baixo],"k")
          plt.plot([direita,esquerda]  , [baixo,baixo],"k")
          cb = plt.colorbar()

          if invertY == True:
              plt.gca().invert_yaxis()
          cb.set_label('Time (s)',fontsize=fontsize)
          cb.ax.tick_params(labelsize=fontsize*0.8)
          plt.xlabel("X (px)",fontsize=fontsize)
          plt.ylabel("Y (px)",fontsize=fontsize)
          plt.xticks(fontsize = fontsize*0.8)
          plt.yticks(fontsize = fontsize*0.8)

          if type(saveName) != type(None):
              plt.savefig(saveName+figformat)

          plt.show()

      else:
          ax.set_aspect('equal')
          for i in range(len(animals)):
            plot = ax.scatter(animals_data[animals[i]+"_x"], animals_data[animals[i]+"_y"], c=c[i], cmap=cmap, s=3)
          ax.plot([esquerda,esquerda] , [baixo,cima],"k")
          ax.plot([esquerda,direita]  , [cima,cima],"k")
          ax.plot([direita,direita]   , [cima,baixo],"k")
          ax.plot([direita,esquerda]  , [baixo,baixo],"k")
          ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)
          ax.set_title(figureTitle, fontsize=fontsize)
          ax.set_xlabel("X (px)", fontsize = fontsize)
          ax.set_ylabel("Y (px)", fontsize = fontsize)

          divider = make_axes_locatable(ax)
          cax = divider.append_axes('right',size='5%', pad=0.05)
          cb = fig.colorbar(plot,cax=cax)
          cb.ax.tick_params(labelsize=fontsize*0.8)
          cb.set_label(label='Time (s)', fontsize=fontsize)

          if invertY == True:
              ax.invert_yaxis()
    
    else:
       for i in range(len(animals)):
        if type(ax) == type(None): 
            plt.figure(figsize=(wSize, hSize), dpi=res)
            plt.title(animals[i], fontsize=fontsize)
            plt.scatter(animals_data[animals[i]+"_x"], animals_data[animals[i]+"_y"], c=c[i], cmap=cmap, s=3)
            plt.plot([esquerda,esquerda] , [baixo,cima],"k")
            plt.plot([esquerda,direita]  , [cima,cima],"k")
            plt.plot([direita,direita]   , [cima,baixo],"k")
            plt.plot([direita,esquerda]  , [baixo,baixo],"k")
            cb = plt.colorbar()

            if invertY == True:
                plt.gca().invert_yaxis()
            cb.set_label('Time (s)',fontsize=fontsize)
            cb.ax.tick_params(labelsize=fontsize*0.8)
            plt.xlabel("X (px)",fontsize=fontsize)
            plt.ylabel("Y (px)",fontsize=fontsize)
            plt.xticks(fontsize = fontsize*0.8)
            plt.yticks(fontsize = fontsize*0.8)

            if type(saveName) != type(None):
                plt.savefig(saveName+animals[i]+figformat)

            plt.show()

        else:
            #TODO: subplots
            ax.set_aspect('equal')
            plot = ax.scatter(animals_data[animals[i]+"_x"], animals_data[animals[i]+"_y"], c=c[i], cmap=cmap, s=3)
            ax.plot([esquerda,esquerda] , [baixo,cima],"k")
            ax.plot([esquerda,direita]  , [cima,cima],"k")
            ax.plot([direita,direita]   , [cima,baixo],"k")
            ax.plot([direita,esquerda]  , [baixo,baixo],"k")
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)
            ax.set_title(animals[i], fontsize=fontsize)
            ax.set_xlabel("X (px)", fontsize = fontsize)
            ax.set_ylabel("Y (px)", fontsize = fontsize)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right',size='5%', pad=0.05)
            cb = fig.colorbar(plot,cax=cax)
            cb.ax.tick_params(labelsize=fontsize*0.8)
            cb.set_label(label='Time (s)', fontsize=fontsize)

            if invertY == True:
                ax.invert_yaxis()


def splitMultiAnimal(data,data_type = '.h5',**kwargs):
    """
    This function is not intended for the end user. _splitMultiAnimal performs 
    the extraction of information from the hdf of multi-animals from the DLC to 
    separate them into different DataFrames. Its output is a dictionary with 
    this data. 

    Parameters
    ----------
    data : pandas DataFrame
        The input tracking data in h5/hdf/csv format (multi animal data).
    data_type : str
        Determine if the data format from DLC is in '.h5' or '.csv' format.
    bodyPart : str, optional
        Body part you want to plot the tracking.
    animals: list, optional
        If you have multi-animal data and want to extract only specific animals, 
        just pass in this variable a list with the name of the designated animal 
        (e.g. animals ['rat1']).
    start : int, optional
        Moment of the video you want tracking to start, in seconds. If the variable 
        is empty (None), the entire video will be processed.
    end : int, optional
        Moment of the video you want tracking to end, in seconds. If the variable is 
        empty (None), the entire video will be processed.
    fps : int
        The recording frames per second.

    Returns
    -------
    out : dict
          The output of this function is a dictionary with the data of each
          animal present in the HDF.

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat

    Notes
    -----
    This function was developed based on DLC multianimal output."""

    import numpy as np
    import pandas as pd

    animals = kwargs.get('animals')
    bodyParts = kwargs.get('bodyParts')
    start= kwargs.get('start')
    end= kwargs.get('end')
    fps= kwargs.get('fps')

    if type(fps) == type(None):
      fps = 30

    animals_data = {}

    if data_type == '.h5':
        if type(animals) == type(None):
            animals = list(set([i[0] for i in list(set(data[data.columns[0][0]].columns))]))

        if type(bodyParts) == type(None):       
            bodyParts = list(set([i[1] for i in list(set(data[data.columns[0][0]].columns))]))

        for i,animal in enumerate(animals):
            parts = {}
            for ç,bodyPart in enumerate(bodyParts):
                temp = data[data.columns[0][0]][animal][bodyPart].iloc[:,0:3]

                if type(start) == type(None) and type(end) == type(None):
                    parts[bodyPart] = ((temp['x'].values).astype(float),
                                      (temp['y'].values).astype(float),
                                      (temp['likelihood'].values).astype(float))
                else:
                    if type(start) == type(None):
                        finish = int(end[i]*fps)
                        parts[bodyPart] = ((temp['x'][None:finish].values).astype(float),
                                          (temp['y'][None:finish].values).astype(float),
                                          (temp['likelihood'][None:finish].values).astype(float))
                    elif type(end) == type(None):
                        init = int(start[i]*fps)
                        parts[bodyPart] = ((temp['x'][init:None].values).astype(float),
                                          (temp['y'][init:None].values).astype(float),
                                          (temp['likelihood'][init:None].values).astype(float))
                    else:     
                        init = int(start[i]*fps)
                        finish = int(end[i]*fps)
                        parts[bodyPart] = ((temp['x'][init:finish].values).astype(float),
                                          (temp['y'][init:finish].values).astype(float),
                                          (temp['likelihood'][init:finish].values).astype(float)) 
                        
            animals_data[animal] = parts
    
    if data_type == '.csv':
        header = [(data[i][0]+' '+data[i][1]) for i in data.columns]
        data.columns = header
        if type(animals) == type(None): 
            animals = list(set(data.iloc[0][1:]))
        if type(bodyParts) == type(None):
            bodyParts = list(set(data.iloc[1][1:])) 

        for i,animal in enumerate(animals):
            parts = {}
            for ç,bodyPart in enumerate(bodyParts):
                if type(start) == type(None) and type(end) == type(None):
                    parts[bodyPart] = ((data[animal+' '+bodyPart].iloc[3:,0]).values.astype(float),
                                      (data[animal+' '+bodyPart].iloc[3:,1]).values.astype(float),
                                      (data[animal+' '+bodyPart].iloc[3:,2]).values.astype(float))
                else:
                    if type(start) == type(None):
                        finish = int(end[i]*fps)
                        parts[bodyPart] = ((data[animal+' '+bodyPart].iloc[3:,0][None:finish]).values.astype(float),
                                          (data[animal+' '+bodyPart].iloc[3:,1][None:finish]).values.astype(float),
                                          (data[animal+' '+bodyPart].iloc[3:,2][None:finish]).values.astype(float))
                    elif type(end) == type(None):
                        init = int(start[i]*fps)
                        parts[bodyPart] = ((data[animal+' '+bodyPart].iloc[3:,0][init:None]).values.astype(float),
                                          (data[animal+' '+bodyPart].iloc[3:,1][init:None]).values.astype(float),
                                          (data[animal+' '+bodyPart].iloc[3:,2][init:None]).values.astype(float))
                    else:     
                        init = int(start[i]*fps)
                        finish = int(end[i]*fps)
                        parts[bodyPart] = ((data[animal+' '+bodyPart].iloc[3:,0][init:finish]).values.astype(float),
                                          (data[animal+' '+bodyPart].iloc[3:,1][init:finish]).values.astype(float),
                                          (data[animal+' '+bodyPart].iloc[3:,2][init:finish]).values.astype(float)) 
                        
            animals_data[animal] = parts      
                
    return animals_data

def multi2single(data,animal,data_type = '.h5',**kwargs):
    """
    This function is used to remove information from a single animal from the 
    h5/hdf file of a DLC multi-animal analysis. The main purpose of this 
    function is to facilitate data analysis, by returning a DataFrame that can 
    be used as input in all PyRAT functions, without the need for any adaptation.
    WARNING: If you run this function and found 'KeyError: 0', just read the data
    again (pd.read_csv(data)).

    Parameters
    ----------
    data : pandas DataFrame
        The input tracking data in h5/hdf format (multi animal data).
    data_type : str
        Determine if the data format from DLC is in h5 or csv format.
    animal : str
        The key of the animal you want to extract from the hdf file. 
        The same name used to label the DLC.
    drop: bool, optional
        If true, will drop the NaN values in the DataFrame.
    bodyPart : str, optional
        Body part you want to plot the tracking.
    start : int, optional
        Moment of the video you want tracking to start, in seconds. If the variable 
        is empty (None), the entire video will be processed.
    end : int, optional
        Moment of the video you want tracking to end, in seconds. If the variable is 
        empty (None), the entire video will be processed.
    fps : int
        The recording frames per second.

    Returns
    -------
    out : DataFrame
          The output of this function is a DataFrame with the data of the animal
          passed in the input.

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat

    Notes
    -----
    This function was developed based on DLC multianimal output."""

    import numpy as np
    import pandas as pd
    # Removed: import pyratlib as rat

    animals = kwargs.get('animals')
    bodyParts = kwargs.get('bodyParts')
    start= kwargs.get('start')
    end= kwargs.get('end')
    fps= kwargs.get('fps')
    drop= kwargs.get('drop')

    if type(drop) == type(None):
        drop = False

    data = splitMultiAnimal(data,
                                data_type=data_type,
                                animals=animals,
                                bodyParts=bodyParts,
                                start=start,
                                end=end,
                                fps=fps)
    
    parts = list(np.repeat(list(data[animal].keys()),3))
    coord = ['x','y','likelihood']*len(list(data[animal].keys()))
    header = ["{}.{}".format(animal, ç) for ç in range(len(parts))]

    df_header = pd.DataFrame([list([-1]+parts),
                             list([0]+coord)], 
                             columns=['coords']+header)
    
    count = 0
    temp_dict = {}

    for ç,part in enumerate(list(data[animal].keys())):
        temp = {header[ç+count]: data[animal][part][0],
                header[ç+1+count]: data[animal][part][1],
                header[ç+2+count]: data[animal][part][2],}
        count +=2
        temp_dict.update(temp)

    df_temp = pd.DataFrame(temp_dict)

    df = pd.concat([df_header, df_temp], ignore_index=True)
    df['coords'] = np.arange(0,len(df['coords']),1)

    if drop:
        df = df.dropna()

    return df

def distance_metrics(data, bodyparts_list,distance=28):
    """
    Returns the distance between the bodyparts.
    
    Parameters
    ----------
    data : pandas DataFrame
        The input tracking data.
    bodyparts_list : list
        List with name of body parts.
    distance : int
        The linkage distance threshold above which, clusters will not be merged.

    Returns
    -------
    d  : array
        High dimension data.
    
    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat
    
    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations."""

    import numpy as np

    values = (data.iloc[2:,1:].values).astype(float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()
    bodyparts = []
    for i in range(len(bodyparts_list)):
      bodyparts.append(np.concatenate(((values[:,lista1.index(bodyparts_list[i]+" - x")]).reshape(1,-1).T,(values[:,lista1.index(bodyparts_list[i]+" - y")]).reshape(1,-1).T), axis=1))
    distances = []
    for k in range(len(bodyparts[0])):
        frame_distances = []
        for i in range(len(bodyparts)):
            distance_row = []
            for j in range( len(bodyparts) ):
                distance_row.append(np.linalg.norm(bodyparts[i][k] - bodyparts[j][k]))
            frame_distances.append(distance_row)
        distances.append(frame_distances)
    distances2 = np.asarray(distances)
    for i in range(len(bodyparts)):
      for k in range(len(bodyparts)):
          distances2[:, i, j] = distances2[:, i, j]/np.max(distances2[:, i, j])
    dist = []
    for i in range(distances2.shape[0]):
        dist.append(distances2[i, np.triu_indices(len(bodyparts), k = 1)[0], np.triu_indices(len(bodyparts), k = 1)[1]])
    
    return dist

def model_distance(dimensions = 2,distance=28,n_jobs=None,verbose=None, perplexity=None,learning_rate=None):
    """
    Returns an array with the cluster by frame, an array with the embedding data in low-dimensional 
    space and the clusterization model.
    
    Parameters
    ----------
    dimensions : int
        Dimension of the embedded space.
    distance : int
        The linkage distance threshold above which, clusters will not be merged.
    n_jobs : int, optional
        The number of parallel jobs to run for neighbors search.
    verbose : int, optional
        Verbosity level.
    perplexity : float, optional
        The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity.
    learning_rate : float, optional
        t-SNE learning rate.

    Returns
    -------
    model : Obj
        AgglomerativeClustering model.
    embedding : Obj
        TSNE embedding
    
    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat
    
    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations."""

    from sklearn.manifold import TSNE
    from sklearn.cluster import AgglomerativeClustering 

    model = AgglomerativeClustering(n_clusters=None,distance_threshold=distance)

    embedding = TSNE(n_components=dimensions,
                     n_jobs=n_jobs,
                     verbose=verbose,
                     perplexity=perplexity,
                     random_state = 42,
                     n_iter = 5000,
                     learning_rate =learning_rate,
                     init='pca',
                     early_exaggeration =12)

    return model, embedding

def ClassifyBehaviorMultiVideos(data, bodyparts_list, dimensions = 2,distance=28, **kwargs):
    """
    Returns an array with the cluster by frame, an array with the embedding data in low-dimensional 
    space and the clusterization model.
    
    Parameters
    ----------
    data : dict with DataFrames
        The input tracking data concatenated.
    bodyparts_list : list
        List with name of body parts.
    dimensions : int
        Dimension of the embedded space.
    distance : int
        The linkage distance threshold above which, clusters will not be merged.
    n_jobs : int, optional
        The number of parallel jobs to run for neighbors search.
    verbose : int, optional
        Verbosity level.
    perplexity : float, optional
        The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity.
    learning_rate : float, optional
        t-SNE learning rate.

    Returns
    -------
    cluster_df : df
        Array with the cluster by frame/video.
    cluster_coord : DataFrame
        Embedding of the training data in low-dimensional space.
    fitted_model : Obj
        AgglomerativeClustering model.

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat
    
    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations."""

    import numpy as np
    import pandas as pd
    # Removed: import pyratlib as rat
    from sklearn.preprocessing import StandardScaler

    n_jobs = kwargs.get('n_jobs')
    verbose = kwargs.get('verbose')
    perplexity = kwargs.get("perplexity")
    learning_rate = kwargs.get("learning_rate")

    if type(n_jobs) == type(None):
      n_jobs=-1
    if type(verbose) == type(None):
      verbose=0
    if type(perplexity) == type(None):
      perplexity = data[next(iter(data))].shape[0]//100
    if type(learning_rate) == type(None):
      learning_rate = (data[next(iter(data))].shape[0]//12)/4

    distancias       = {}
    dist_scaled      = {}
    cluster_labels   = {}
    distance_df      = {}

    model,embedding  = model_distance(dimensions = dimensions,
                                          distance=distance,
                                          n_jobs=n_jobs,
                                          verbose=verbose, 
                                          perplexity=perplexity,
                                          learning_rate=learning_rate)


    for i,video in enumerate(data):
        dist_temp            = np.asarray(distance_metrics(data[video],
                                          bodyparts_list=bodyparts_list,
                                          distance=distance))    
        distancias[video]    = dist_temp
        dist_scaled[video]   = StandardScaler().fit_transform(distancias[video])


    dist_scaled_all    = np.concatenate([dist_scaled[x] for x in dist_scaled], 0)
    X_transformed      = embedding.fit_transform(dist_scaled_all)
    fitted_model       = model.fit(dist_scaled_all)
    cluster_labels_all = model.labels_


    for i,video in enumerate(data): 
        if i == 0:
            cluster_labels[video] = cluster_labels_all[0:dist_scaled[video].shape[0]]
            index0                = dist_scaled[video].shape[0]
        else:
            cluster_labels[video] = cluster_labels_all[index0:(index0+dist_scaled[video].shape[0])]
            index0                = index0+dist_scaled[video].shape[0]


    cluster_coord                      = pd.DataFrame.from_dict({ 'x_n_samples':X_transformed[:,0],'y_n_components':X_transformed[:,1] })
    distance_df                        = pd.DataFrame.from_dict({'distance '+str(dist): dist_scaled_all[:,dist] for dist in range(dist_scaled_all.shape[1])})
    cluster_coord[distance_df.columns] = distance_df
    cluster_df                         = pd.DataFrame.from_dict(cluster_labels)


    return cluster_df, cluster_coord ,fitted_model

def dendrogram(model, **kwargs):
    from scipy.cluster.hierarchy import dendrogram

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    dendrogram(linkage_matrix, truncate_mode="level", p=3,leaf_rotation=90, leaf_font_size=10)

def ClassifyBehavior(data,video, bodyparts_list, dimensions = 2,distance=28,**kwargs):
    """
    Returns an array with the cluster by frame, an array with the embedding data in low-dimensional 
    space and the clusterization model.
    
    Parameters
    ----------
    data : pandas DataFrame
        The input tracking data.
    video : str
        Video directory
    bodyparts_list : list
        List with name of body parts.
    dimensions : int
        Dimension of the embedded space.
    distance : int
        The linkage distance threshold above which, clusters will not be merged.
    startIndex : int, optional
        Initial index.
    endIndex : int, optional
        Last index.
    n_jobs : int, optional
        The number of parallel jobs to run for neighbors search.
    verbose : int, optional
        Verbosity level.
    perplexity : float, optional
        The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity.
    learning_rate : float, optional
        t-SNE learning rate.
    directory : str, optional
        Path where frame images will be saved.
    return_metrics : bool, optional
         Where True, returns t-SNE metrics, otherwise does not return t-SNE metrics.
    knn_n_neighbors : int, optional
        Number of neighbors to use by default for kneighbors queries in KNN metric.
    knc_n_neighbors : int, optional
        Number of neighbors to use by default for kneighbors queries in KNC metric.
    n : int, optional
        Number of N randomly chosen points in CPD metric.
    Returns
    -------
    cluster_labels : array
        Array with the cluster by frame.
    X_transformed : array
        Embedding of the training data in low-dimensional space.
    model : Obj
        AgglomerativeClustering model.
    d  : array
        High dimension data.
    knn : int, optional
        The fraction of k-nearest neighbours in the original highdimensional data that are preserved as k-nearest neighbours in the embedding.
    knc : int, optional
        The fraction of k-nearest class means in the original data that are preserved as k-nearest class means in the embedding. This is computed for class means only and averaged across all classes.
    cpd : Obj, optional
        Spearman correlation between pairwise distances in the high-dimensional space and in the embedding.
    
    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat
    
    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations."""
    from sklearn.manifold import TSNE
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    import os
    import cv2
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from scipy import stats
    startIndex = kwargs.get('startIndex')
    endIndex = kwargs.get('endIndex')
    n_jobs = kwargs.get('n_jobs')
    verbose = kwargs.get('verbose')
    perplexity = kwargs.get("perplexity")
    learning_rate = kwargs.get("learning_rate")
    directory = kwargs.get("directory")
    return_metrics = kwargs.get("return_metrics")
    knn_n_neighbors = kwargs.get("knn_n_neighbors")
    knc_n_neighbors = kwargs.get("knc_n_neighbors")
    n = kwargs.get("n")
    k = 1
    if type(startIndex) == type(None):
      startIndex = 0
    if type(endIndex) == type(None):
      endIndex = data.shape[0]-3
    if type(n_jobs) == type(None):
      n_jobs=-1
    if type(verbose) == type(None):
      verbose=0
    if type(perplexity) == type(None):
      perplexity = data[startIndex:endIndex].shape[0]//100
    if type(learning_rate) == type(None):
      learning_rate = (data[startIndex:endIndex].shape[0]//12)/4
    if type(directory) == type(None):
      directory = os.getcwd()
    if type(return_metrics) == type(None):
      return_metrics == 0

    directory=directory+os.sep+"images"
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
    
    values = (data.iloc[2:,1:].values).astype(float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()
    bodyparts = []
    for i in range(len(bodyparts_list)):
      bodyparts.append(np.concatenate(((values[:,lista1.index(bodyparts_list[i]+" - x")]).reshape(1,-1).T,(values[:,lista1.index(bodyparts_list[i]+" - y")]).reshape(1,-1).T), axis=1))
    distances = []
    for k in range(len(bodyparts[0])):
        frame_distances = []
        for i in range(len(bodyparts)):
            distance_row = []
            for j in range( len(bodyparts) ):
                distance_row.append(np.linalg.norm(bodyparts[i][k] - bodyparts[j][k]))
            frame_distances.append(distance_row)
        distances.append(frame_distances)
    distances2 = np.asarray(distances)
    for i in range(len(bodyparts)):
      for k in range(len(bodyparts)):
          distances2[:, i, j] = distances2[:, i, j]/np.max(distances2[:, i, j])
    d = []
    for i in range(distances2.shape[0]):
        d.append(distances2[i, np.triu_indices(len(bodyparts), k = 1)[0], np.triu_indices(len(bodyparts), k = 1)[1]])
    
    d = StandardScaler().fit_transform(d)
    embedding = TSNE(n_components=dimensions, n_jobs=n_jobs, verbose=verbose, perplexity=perplexity, random_state = 42, n_iter = 5000, learning_rate=learning_rate, init = "pca", early_exaggeration = 12)
    X_transformed = embedding.fit_transform(d[startIndex:endIndex])
    model = AgglomerativeClustering(n_clusters=None,distance_threshold=distance)
    model = model.fit(d[startIndex:endIndex])
    cluster_labels = model.labels_   
    frames = data.scorer[2:].values.astype(int)
    for i in np.unique(cluster_labels):
      os.makedirs(directory+os.sep+"cluster"+ str(i))
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    position = (10,50)
    ind = 0
    while success:
      if (np.isin(count, frames[startIndex:endIndex])):
          a = cv2.imwrite(directory+os.sep+"cluster"+str(model.labels_[ind])+os.sep+"frame%d.jpg" % count, image)
          ind = ind +1 
      success,image = vidcap.read()
      count += 1
    for i in np.unique(cluster_labels):
      plt.bar(i, cluster_labels[cluster_labels==i].shape, color = "C0")
    plt.xticks(np.arange(model.n_clusters_))
    plt.xlabel("Clusters")
    plt.ylabel("Frames")
    plt.show()
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    plt.figure(figsize=(10,5))
    plt.title("Hierarchical Clustering Dendrogram")
    dendrogram(linkage_matrix, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    fig, ax = plt.subplots(figsize=(10,10), dpi =80)
    i = 0
    color = plt.cm.get_cmap("rainbow", model.n_clusters_)
    for x in range(model.n_clusters_):
      sel = cluster_labels == x
      
      pontos = ax.scatter(X_transformed[sel,0], X_transformed[sel,1], label=str(x), s=1, color = color(i))
      i = i+1
    plt.legend()
    plt.title('Clusters')
    plt.show()
    if return_metrics == 1:
      if type(knn_n_neighbors) == type(None):
        knn_n_neighbors = model.n_clusters_//2
      if type(knc_n_neighbors) == type(None):
        knc_n_neighbors = model.n_clusters_//2
      if type(n) == type(None):
        n = 1000
      data_HDim = d[startIndex:endIndex]
      data_emb = X_transformed
      neigh = NearestNeighbors(n_neighbors=knn_n_neighbors)
      neigh.fit(data_HDim)
      neigh2 = NearestNeighbors(n_neighbors=knn_n_neighbors)
      neigh2.fit(data_emb)
      intersections = 0.0
      for i in range(len(data_HDim)):
        intersections += len(set(neigh.kneighbors(data_HDim, return_distance = False)[i]) & set(neigh2.kneighbors(data_emb, return_distance = False)[i]))
      knn = intersections / len(data_HDim) / knn_n_neighbors
      clusters =  len(np.unique(cluster_labels))
      clusters_HDim = np.zeros((clusters,data_HDim.shape[1]))
      clusters_tsne = np.zeros((clusters,data_emb.shape[1]))
      for i in np.unique(cluster_labels):
        clusters_HDim[i,:] = np.mean(data_HDim[np.unique(cluster_labels, return_inverse=True)[1] == np.unique(cluster_labels, return_inverse=True)[0][i], :], axis = 0)
        clusters_tsne[i,:] = np.mean(data_emb[np.unique(cluster_labels, return_inverse=True)[1] == np.unique(cluster_labels, return_inverse=True)[0][i], :], axis = 0)
      neigh = NearestNeighbors(n_neighbors=knc_n_neighbors)
      neigh.fit(clusters_HDim)
      neigh2 = NearestNeighbors(n_neighbors=knc_n_neighbors)
      neigh2.fit(clusters_tsne)
      intersections = 0.0
      for i in range(clusters):
        intersections += len(set(neigh.kneighbors(clusters_HDim, return_distance = False)[i]) & set(neigh2.kneighbors(clusters_tsne, return_distance = False)[i]))
      knc = intersections / clusters / knc_n_neighbors
      dist_alto = np.zeros(n)
      dist_tsne = np.zeros(n)
      for i in range(n):
        a = np.random.randint(0,len(data_HDim), size = 1)
        b = np.random.randint(0,len(data_HDim), size = 1)
        dist_alto[i] = np.linalg.norm(data_HDim[a] - data_HDim[b])
        dist_tsne[i] = np.linalg.norm(data_emb[a] - data_emb[b])
      cpd = stats.spearmanr(dist_alto, dist_tsne)
      return cluster_labels, data_emb, model, data_HDim, knn, knc, cpd
    else:
      return cluster_labels, X_transformed, model, d[startIndex:endIndex]


def Trajectory(data,bodyPart,bodyPartBox = None, **kwargs):
    saveName= kwargs.get('saveName')
    start= kwargs.get('start')
    end= kwargs.get('end')
    figureTitle = kwargs.get('figureTitle')
    fps = kwargs.get('fps')
    ax = kwargs.get('ax')
    limit_boundaries = kwargs.get('limit_boundaries')
    xLimMin = kwargs.get('xLimMin')
    xLimMax = kwargs.get('xLimMax')
    yLimMin = kwargs.get('yLimMin')
    yLimMax = kwargs.get('yLimMax')

    if type(limit_boundaries) == type(None):
      limit_boundaries = False
    fig = kwargs.get('fig')
    if fig is None and ax is not None:
        fig = ax.figure
    if type(fps) == type(None):
      fps = 30
    cmapType = kwargs.get('cmapType')
    if type(cmapType) == type(None):
      cmapType = 'viridis'
    hSize = kwargs.get('hSize')
    if type(hSize) == type(None):
      hSize = 6
    wSize = kwargs.get('wSize')
    if type(wSize) == type(None):
      wSize = 8
    bins = kwargs.get('bins')
    if type(bins) == type(None):
      bins = 30
    fontsize = kwargs.get('fontsize')
    if type(fontsize) == type(None):
      fontsize = 15
    invertY = kwargs.get('invertY')
    if type(invertY) == type(None):
      invertY = True
    figformat = kwargs.get('figformat')
    if type(figformat) == type(None):
      figformat = '.eps'
    res = kwargs.get('res')
    if type(res) == type(None):
      res = 80  

    values = (data.iloc[2:,1:].values).astype(float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()

    if type(start) == type(None):
        x = values[:,lista1.index(bodyPart+" - x")]
        y = values[:,lista1.index(bodyPart+" - y")]
    else:
        init = int(start*fps)
        finish = int(end*fps)
        x = values[:,lista1.index(bodyPart+" - x")][init:finish]
        y = values[:,lista1.index(bodyPart+" - y")][init:finish]


    cmap = plt.get_cmap(cmapType)


    if type(bodyPartBox) == type(None):
      c = np.linspace(0, x.size/fps, x.size)
      esquerda = xLimMin
      direita = xLimMax
      baixo = yLimMin
      cima = yLimMax
    else:
      c = np.linspace(0, x.size/fps, x.size)
      esquerda = values[:,lista1.index(bodyPartBox+" - x")].min()
      direita = values[:,lista1.index(bodyPartBox+" - x")].max()
      baixo = values[:,lista1.index(bodyPartBox+" - y")].min()
      cima = values[:,lista1.index(bodyPartBox+" - y")].max()

    if limit_boundaries:
        testeX = []
        for i in range(len(x)):
            if x[i] >= direita:
                testeX.append(direita)
            elif x[i] <= esquerda:
                testeX.append(esquerda)
            else:
                testeX.append(x[i])
        
        testeY = []
        for i in range(len(x)):
            if y[i] >= cima:
                testeY.append(cima)
            elif y[i] <= baixo:
                testeY.append(baixo)
            else:
                testeY.append(y[i])
    else:
        testeX = x
        testeY = y

    if type(ax) == type(None): 
        plt.figure(figsize=(wSize, hSize), dpi=res)
        plt.title(figureTitle, fontsize=fontsize)
        plt.scatter(testeX, testeY, c=c, cmap=cmap, s=3)
        plt.plot([esquerda,esquerda] , [baixo,cima],"k")
        plt.plot([esquerda,direita]  , [cima,cima],"k")
        plt.plot([direita,direita]   , [cima,baixo],"k")
        plt.plot([direita,esquerda]  , [baixo,baixo],"k")
        cb = plt.colorbar()

        if invertY == True:
            plt.gca().invert_yaxis()
        cb.set_label('Time (s)',fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize*0.8)
        plt.xlabel("X (px)",fontsize=fontsize)
        plt.ylabel("Y (px)",fontsize=fontsize)
        plt.xticks(fontsize = fontsize*0.8)
        plt.yticks(fontsize = fontsize*0.8)

        if type(saveName) != type(None):
            plt.savefig(saveName+figformat)

        plt.show()

    else:
        ax.set_aspect('equal')
        plot = ax.scatter(testeX, testeY, c=c, cmap=cmap, s=3)
        ax.plot([esquerda,esquerda] , [baixo,cima],"k")
        ax.plot([esquerda,direita]  , [cima,cima],"k")
        ax.plot([direita,direita]   , [cima,baixo],"k")
        ax.plot([direita,esquerda]  , [baixo,baixo],"k")
        ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)
        ax.set_title(figureTitle, fontsize=fontsize)
        ax.set_xlabel("X (px)", fontsize = fontsize)
        ax.set_ylabel("Y (px)", fontsize = fontsize)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right',size='5%', pad=0.05)
        cb = fig.colorbar(plot,cax=cax)
        cb.ax.tick_params(labelsize=fontsize*0.8)
        cb.set_label(label='Time (s)', fontsize=fontsize)

        if invertY == True:
            ax.invert_yaxis()

def Heatmap(data, bodyPart, **kwargs):
    saveName= kwargs.get('saveName')
    start= kwargs.get('start')
    end= kwargs.get('end')
    figureTitle = kwargs.get('figureTitle')
    fps = kwargs.get('fps')
    ax = kwargs.get('ax')
    fig = kwargs.get('fig')
    if fig is None and ax is not None:
        fig = ax.figure
    bodyPartBox = kwargs.get('bodyPartBox')
    limit_boundaries = kwargs.get('limit_boundaries')
    xLimMin = kwargs.get('xLimMin')
    xLimMax = kwargs.get('xLimMax')
    yLimMin = kwargs.get('yLimMin')
    yLimMax = kwargs.get('yLimMax')

    if type(limit_boundaries) == type(None):
      limit_boundaries = False
    if type(fps) == type(None):
      fps = 30
    if type(bodyPartBox) == type(None):
      bodyPartBox = bodyPart
    cmapType = kwargs.get('cmapType')
    if type(cmapType) == type(None):
      cmapType = 'viridis'
    hSize = kwargs.get('hSize')
    if type(hSize) == type(None):
      hSize = 6
    wSize = kwargs.get('wSize')
    if type(wSize) == type(None):
      wSize = 8
    bins = kwargs.get('bins')
    if type(bins) == type(None):
      bins = 30
    fontsize = kwargs.get('fontsize')
    if type(fontsize) == type(None):
      fontsize = 15
    invertY = kwargs.get('invertY')
    if type(invertY) == type(None):
      invertY = True
    figformat = kwargs.get('figformat')
    if type(figformat) == type(None):
      figformat = '.eps'
    vmax = kwargs.get('vmax')
    if type(vmax) == type(None):
      vmax = 1000
    res = kwargs.get('res')
    if type(res) == type(None):
      res = 80  

    values = (data.iloc[2:,1:].values).astype(float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()

    esquerda = values[:,lista1.index(bodyPartBox+" - x")].min()
    direita = values[:,lista1.index(bodyPartBox+" - x")].max()
    baixo = values[:,lista1.index(bodyPartBox+" - y")].min()
    cima = values[:,lista1.index(bodyPartBox+" - y")].max()

    if type(start) == type(None):
        x = values[:,lista1.index(bodyPart+" - x")]
        y = values[:,lista1.index(bodyPart+" - y")]
    else:
        init = int(start*fps)
        finish = int(end*fps)
        x = values[:,lista1.index(bodyPart+" - x")][init:finish]
        y = values[:,lista1.index(bodyPart+" - y")][init:finish]

    if limit_boundaries:
        xx = []
        for i in range(len(x)):
            if x[i] >= direita:
                xx.append(direita)
            elif x[i] <= esquerda:
                xx.append(esquerda)
            else:
                xx.append(x[i])
        
        yy = []
        for i in range(len(x)):
            if y[i] >= cima:
                yy.append(cima)
            elif y[i] <= baixo:
                yy.append(baixo)
            else:
                yy.append(y[i])
    else:
        xx = x
        yy = y
    
    if type(ax) == type(None):
        plt.figure(figsize=(wSize, hSize), dpi=res)

        if type(xLimMin) != type(None):
            plt.hist2d(xx,yy, bins = bins, vmax = vmax,cmap=plt.get_cmap(cmapType), range=[[xLimMin,xLimMax],[yLimMin,yLimMax]])
        else:
            plt.hist2d(xx,yy, bins = bins, vmax = vmax,cmap=plt.get_cmap(cmapType))

        cb = plt.colorbar()

        plt.title(figureTitle, fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize*0.8)
        plt.xlabel("X (px)",fontsize=fontsize)
        plt.ylabel("Y (px)",fontsize=fontsize)
        plt.xticks(fontsize = fontsize*0.8)
        plt.yticks(fontsize = fontsize*0.8)
        if invertY == True:
            plt.gca().invert_yaxis()

        if type(saveName) != type(None):
            plt.savefig(saveName+figformat)

        plt.show()
    else:
        if type(xLimMin) != type(None):
            ax.hist2d(xx,yy, bins = bins, vmax = vmax,cmap=plt.get_cmap(cmapType), range=[[xLimMin,xLimMax],[yLimMin,yLimMax]])
        else:
            ax.hist2d(xx,yy, bins = bins, vmax = vmax,cmap=plt.get_cmap(cmapType))
        ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)
        ax.set_title(figureTitle, fontsize=fontsize)
        ax.set_xlabel("X (px)", fontsize = fontsize)
        ax.set_ylabel("Y (px)", fontsize = fontsize)
        if invertY == True:
            ax.invert_yaxis()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right',size='5%', pad=0.05)

        im = ax.imshow([xx,yy], cmap=plt.get_cmap(cmapType))
        cb = fig.colorbar(im,cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fontsize*0.8)

def pixel2centimeters(data, pixel_max,pixel_min,max_real, min_real=0):
    return min_real + ((data-pixel_min)/(pixel_max-pixel_min)) * (max_real-min_real)

def MotionMetrics (data,bodyPart,filter=1,fps=30,max_real=60,min_real=0):
    values = (data.iloc[2:,1:].values).astype(float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()

    dataX = values[:,lista1.index(bodyPart+" - x")]
    dataY = values[:,lista1.index(bodyPart+" - y")]

    dataX = pixel2centimeters(dataX,dataX.max(),dataX.min(), max_real,0)
    dataY = pixel2centimeters(dataY,dataY.max(),dataY.min(), min_real,0)

    time = np.arange(0,((1/fps)*len(dataX)), (1/fps))
    df = pd.DataFrame(time/60, columns = ["Time"])
    dist = np.hypot(np.diff(dataX, prepend=dataX[0]), np.diff(dataY, prepend=dataY[0]))
    dist[dist>=filter] = 0
    dist[0] = "nan"
    df["Distance"] = dist
    df['Speed'] = df['Distance']/(1/fps)
    df['Acceleration'] =  df['Speed'].diff().abs()/(1/fps)

    return df

def FieldDetermination(Fields=1,plot=False,**kwargs):
    ax = kwargs.get('ax')
    ret = kwargs.get('ret')
    posit = kwargs.get('posit')
    data = kwargs.get('data')
    bodyPartBox = kwargs.get('bodyPartBox')
    invertY = kwargs.get('invertY')
    if type(invertY) == type(None):
      invertY = True
    obj_color = kwargs.get('obj_color')
    if type(obj_color) == type(None):
      obj_color = 'r'
    if type(ret) == type(None):
      ret = True

    null = 0
    fields = pd.DataFrame(columns=['fields','center_x','center_y', 'radius', 'a_x', 'a_y' , 'height', 'width'])
    circle = []
    rect = []
    if plot:
        values = (data.iloc[2:,1:].values).astype(float)
        lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()
        ax = plt.gca()
        esquerda = values[:,lista1.index(bodyPartBox+" - x")].min()
        direita = values[:,lista1.index(bodyPartBox+" - x")].max()
        baixo = values[:,lista1.index(bodyPartBox+" - y")].min()
        cima = values[:,lista1.index(bodyPartBox+" - y")].max()

    if type(posit) == type(None):
        for i in range(Fields):
            print('Enter the object type '+ str(i+1) + " (0 - circular, 1 - rectangular):")
            objectType = int(input())
            if objectType == 0:
                print('Enter the X value of the center of the field ' + str(i+1) + ':')
                centerX = int(input())
                print('Enter the Y value of the center of the field ' + str(i+1) + ':')
                centerY = int(input())
                print('Enter the radius value of the field ' + str(i+1) + ':')
                radius = int(input())
                circle.append(plt.Circle((centerX, centerY), radius, color=obj_color,fill = False))
                df2 = pd.DataFrame([[objectType, centerX, centerY,radius,null,null,null,null]], columns=['fields','center_x','center_y', 'radius', 'a_x', 'a_y' , 'height', 'width'])
            else:
                print('Enter the X value of the field\'s lower left vertex ' + str(i+1) + ':')
                aX = int(input())
                print('Enter the Y value of the field\'s lower left vertex ' + str(i+1) + ':')
                aY = int(input())
                print('Enter the field height value ' + str(i+1) + ':')
                height = int(input())
                print('Enter the field\'s width value ' + str(i+1) + ':')
                width = int(input())
                rect.append(patches.Rectangle((aX, aY), height, width, linewidth=1, edgecolor=obj_color, facecolor='none'))
                df2 = pd.DataFrame([[objectType, null,null, null ,aX,aY,height,width]], columns=['fields','center_x','center_y', 'radius', 'a_x', 'a_y' , 'height', 'width'])
            fields = fields.append(df2, ignore_index=True)
    else:
        for i,v in enumerate(posit):
            df2 = pd.DataFrame([[posit[v][0], posit[v][1], posit[v][2],posit[v][3],posit[v][4],posit[v][5],posit[v][6],posit[v][7]]], 
                                 columns=['fields','center_x','center_y', 'radius', 'a_x', 'a_y','height', 'width'])
            if posit[v][0] == 1:
                rect.append(patches.Rectangle((float(posit[v][4]),float(posit[v][5])), float(posit[v][6]), float(posit[v][7]), linewidth=1, edgecolor=obj_color, facecolor='none'))
            if posit[v][0] == 0:
                circle.append(plt.Circle((float(posit[v][1]),float(posit[v][2])), float(posit[v][3]), color=obj_color,fill = False))
            fields = fields.append(df2, ignore_index=True)

    if plot:
        ax.plot([esquerda,esquerda] , [baixo,cima],"k")
        ax.plot([esquerda,direita]  , [cima,cima],"k")
        ax.plot([direita,direita]   , [cima,baixo],"k")
        ax.plot([direita,esquerda]  , [baixo,baixo],"k")
        if invertY == True:
            ax.invert_yaxis()
        for i in range(len(circle)):
            ax.add_patch(circle[i])
        for i in range(len(rect)):
            ax.add_patch(rect[i])

    if ret:
            return fields

def Interaction(data,bodyPart,fields,fps=30):
    values = (data.iloc[2:,1:].values).astype(float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()

    dataX = values[:,lista1.index(bodyPart+" - x")]
    dataY = values[:,lista1.index(bodyPart+" - y")]

    numObjects = len(fields.index)
    interact = np.zeros(len(dataX))

    for i in range(len(interact)):
        for j in range(numObjects):
            if fields['fields'][0] == 0:
                if ((dataX[i] - fields['center_x'][j])**2 + (dataY[i] - fields['center_y'][j])**2 <= fields['radius'][j]**2):
                    interact[i] = j +1
            else:
                if fields['a_x'][j] <= dataX[i] <= (fields['a_x'][j] + fields['height'][j]) and fields['a_y'][j] <= dataY[i] <= (fields['a_y'][j] + fields['width'][j]):
                    interact[i] = j +1

        interactsDf = pd.DataFrame(columns=['start','end','obj'])

    obj = 0
    start = 0
    end = 0
    fps =fps

    for i in range(len(interact)):
        if obj != interact[i]:
            end = ((i-1)/fps)
            df = pd.DataFrame([[start,end,obj]],columns=['start','end','obj'])
            obj =  interact[i]
            start = end
            interactsDf = interactsDf.append(df, ignore_index=True)

    start = end
    end = (len(interact)-1)/fps
    obj = interact[-1]
    df = pd.DataFrame([[start,end,obj]],columns=['start','end','obj'])
    interactsDf = interactsDf.append(df, ignore_index=True)

    return interactsDf, interact

def Reports(df_list,list_name,bodypart,fields=None,filter=0.3,fps=30):
    relatorio = pd.DataFrame(columns=['file','video time (min)','dist (cm)', 'speed (cm/s)'])
  
    if type(fields) != type(None):  
        for i in range(len(fields)):
            relatorio["field_{0}".format(i+1)] = []
            relatorio["time_field_{0}".format(i+1)] = []

    for i,v in enumerate(df_list):
        lista = [list_name[i]]

        DF = MotionMetrics(df_list[i], bodypart, filter=filter, fps=fps)

        time = DF.Time.iloc[-1]
        dist = DF.Distance.sum()
        vMedia = DF.Speed.mean()
        
        lista.append(time)
        lista.append(dist)
        lista.append(vMedia)
        if type(fields) != type(None): 
            interacts,_ = Interaction(df_list[i], bodypart, fields, fps = fps)
            for i in range(len(fields)):
                lista.append(interacts["obj"][interacts["obj"] == i+1].count())
                lista.append((interacts["end"][interacts["obj"] == i+1]-interacts["start"][interacts["obj"] == i+1]).sum())
        relatorio_temp = pd.DataFrame([lista], columns=relatorio.columns)
        relatorio = relatorio.append(relatorio_temp, ignore_index=True)

    return relatorio

def DrawLine(x, y, angle, **kwargs):
    ax = kwargs.get('ax')
    arrow_color = kwargs.get('arrow_color')
    arrow_width = kwargs.get('arrow_width')
    if type(arrow_width) == type(None):
      arrow_width = 2
    head_width = kwargs.get('head_width')
    if type(head_width) == type(None):
      head_width = 7
    arrow_size = kwargs.get('arrow_size')
    if type(arrow_size) == type(None):
      arrow_size = 10

    if type(ax) == type(None):
        return plt.arrow(x, y, arrow_size*np.cos(angle), arrow_size*np.sin(angle),width = arrow_width,head_width=head_width,fc = arrow_color)
    else:
        return ax.arrow(x, y, arrow_size*np.cos(angle), arrow_size*np.sin(angle),width = arrow_width,head_width=head_width,fc = arrow_color)

def HeadOrientation(data, step, head = None, tail = None, **kwargs):
    ax = kwargs.get('ax')
    start= kwargs.get('start')
    end= kwargs.get('end')
    figureTitle = kwargs.get('figureTitle')
    saveName = kwargs.get('saveName')
    hSize = kwargs.get('hSize')
    bodyPartBox = kwargs.get('bodyPartBox')
    arrow_color = kwargs.get('arrow_color')
    limit_boundaries = kwargs.get('limit_boundaries')
    xLimMin = kwargs.get('xLimMin')
    xLimMax = kwargs.get('xLimMax')
    yLimMin = kwargs.get('yLimMin')
    yLimMax = kwargs.get('yLimMax')
    if type(limit_boundaries) == type(None):
      limit_boundaries = False
    if type(bodyPartBox) == type(None):
      bodyPartBox = tail
    fps = kwargs.get('fps')
    if type(fps) == type(None):
      fps = 30
    res = kwargs.get('res')
    if type(res) == type(None):
      res = 80  
    if type(hSize) == type(None):
      hSize = 6
    wSize = kwargs.get('wSize')
    if type(wSize) == type(None):
      wSize = 8
    fontsize = kwargs.get('fontsize')
    if type(fontsize) == type(None):
      fontsize = 15
    invertY = kwargs.get('invertY')
    if type(invertY) == type(None):
      invertY = True
    figformat = kwargs.get('figformat')
    if type(figformat) == type(None):
      figformat = '.eps'
    arrow_width = kwargs.get('arrow_width')
    if type(arrow_width) == type(None):
      arrow_width = 2
    head_width = kwargs.get('head_width')
    if type(head_width) == type(None):
      head_width = 7 
    arrow_size = kwargs.get('arrow_size')
    if type(arrow_size) == type(None):
      arrow_size = 10

    values = (data.iloc[2:,1:].values).astype(float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()

    if type(start) == type(None):
        tailX = values[:,lista1.index(tail+" - x")] 
        tailY = values[:,lista1.index(tail+" - y")]

        cervicalX = values[:,lista1.index(head+" - x")]
        cervicalY = values[:,lista1.index(head+" - y")]
    else:
        init = int(start*fps)
        finish = int(end*fps)

        tailX = values[:,lista1.index(tail+" - x")][init:finish] 
        tailY = values[:,lista1.index(tail+" - y")][init:finish]

        cervicalX = values[:,lista1.index(head+" - x")][init:finish]
        cervicalY = values[:,lista1.index(head+" - y")][init:finish]

    boxX = values[:,lista1.index(bodyPartBox+" - x")]
    boxY = values[:,lista1.index(bodyPartBox+" - y")]

    if type(bodyPartBox) == type(None):
      
      esquerda = xLimMin
      direita = xLimMax
      baixo = yLimMin
      cima = yLimMax
    else:
      
      esquerda = values[:,lista1.index(bodyPartBox+" - x")].min()
      direita = values[:,lista1.index(bodyPartBox+" - x")].max()
      baixo = values[:,lista1.index(bodyPartBox+" - y")].min()
      cima = values[:,lista1.index(bodyPartBox+" - y")].max()

    if limit_boundaries:
        testeX = []
        for i in range(len(tailX)):
            if tailX[i] >= direita:
                testeX.append(direita)
            elif tailX[i] <= esquerda:
                testeX.append(esquerda)
            else:
                testeX.append(tailX[i])
        
        testeY = []
        for i in range(len(tailY)):
            if tailY[i] >= cima:
                testeY.append(cima)
            elif tailY[i] <= baixo:
                testeY.append(baixo)
            else:
                testeY.append(tailY[i])
    else:
        testeX = tailX
        testeY = tailY

    if limit_boundaries:
        tX = []
        for i in range(len(cervicalX)):
            if cervicalX[i] >= direita:
                tX.append(direita)
            elif cervicalX[i] <= esquerda:
                tX.append(esquerda)
            else:
                tX.append(cervicalX[i])
        
        tY = []
        for i in range(len(cervicalY)):
            if cervicalY[i] >= cima:
                tY.append(cima)
            elif cervicalY[i] <= baixo:
                tY.append(baixo)
            else:
                tY.append(cervicalY[i])
    else:
        tX = cervicalX
        tY = cervicalY

    rad = np.arctan2((np.asarray(tY) - np.asarray(testeY)),(np.asarray(tX) - np.asarray(testeX)))

    if type(ax) == type(None):
        plt.figure(figsize=(wSize, hSize), dpi=res)
        plt.title(figureTitle, fontsize=fontsize)
        plt.gca().set_aspect('equal')
      
        if invertY == True:
            plt.gca().invert_yaxis()
      
        plt.xlabel("X (px)",fontsize=fontsize)
        plt.ylabel("Y (px)",fontsize=fontsize)
        plt.xticks(fontsize = fontsize*0.8)
        plt.yticks(fontsize = fontsize*0.8)
      
        for i in range(0,len(tailY),step):
            DrawLine(tX[i], tY[i], (rad[i]), ax = ax,arrow_color = arrow_color, arrow_size = arrow_size)
            
        plt.plot([esquerda,esquerda] , [baixo,cima],"k")
        plt.plot([esquerda,direita]  , [cima,cima],"k")
        plt.plot([direita,direita]   , [cima,baixo],"k")
        plt.plot([direita,esquerda]  , [baixo,baixo],"k")

        if type(saveName) != type(None):
            plt.savefig(saveName+figformat)

        plt.show()

    else:
        ax.set_aspect('equal')
        for i in range(0,len(tailY),step):
            DrawLine(tX[i], tY[i], (rad[i]), ax =ax,arrow_color = arrow_color,arrow_size = arrow_size)
        ax.plot([esquerda,esquerda] , [baixo,cima],"k")
        ax.plot([esquerda,direita]  , [cima,cima],"k")
        ax.plot([direita,direita]   , [cima,baixo],"k")
        ax.plot([direita,esquerda]  , [baixo,baixo],"k")
        ax.set_title(figureTitle, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)
        ax.set_xlabel("X (px)", fontsize = fontsize)
        ax.set_ylabel("Y (px)", fontsize = fontsize)
        if invertY == True:
            ax.invert_yaxis()

def SignalSubset(sig_data,freq,fields, **kwargs):
    start_time= kwargs.get('start_time')
    end_time = kwargs.get('end_time')

    if type(fields) == type(None):
        dicts = {}
        if type(start_time) == type(None):
            keys = range(len(end_time))
        else:
            keys = range(len(start_time))
        for j in keys:
            cortes = {}
            for ç,canal in enumerate(sig_data.columns):
                if type(start_time) == type(None):
                    cortes[ç] = sig_data[canal][None:end_time[j]*freq]
                    dicts[j] = cortes 
                elif type(end_time) == type(None):
                    cortes[ç] = sig_data[canal][start_time[j]*freq:None]
                    dicts[j] = cortes 
                else: 
                    cortes[ç] = sig_data[canal][start_time[j]*freq:end_time[j]*freq]
                    dicts[j] = cortes

    else:
        lista = []
        start = []
        end = []

        for i in fields['obj'].unique():
            if i != 0:
                start.append(fields.start.loc[(fields.obj == i)].values)
                end.append(fields.end.loc[(fields.obj == i)].values)
                lista.append((start,end))

        dicts = {}
        keys = range(len(list(fields['obj'].unique())[1:]))

        for j in keys:
            cortes = {}
            for ç,canal in enumerate(sig_data.columns):
                for i in range(len(lista[0][0][j])):
                    cortes[ç] = sig_data[canal][int(lista[0][0][j][i]*freq):int(lista[0][1][j][i]*freq)]
                    dicts[j] = cortes

    return dicts

def LFP(data):
    column_name = []
    if len(data['allad'][0]) == 192:
        time = np.arange(0,len(data['allad'][0][128])/data['adfreq'][0][0],1/data['adfreq'][0][0]) 
        values = np.zeros((len(data['allad'][0][128]),64))
        ç = 128
        for j in range(64):
            for r in range(len(data['allad'][0][128])):
                values[r][j] = data['allad'][0][ç][r]
            column_name.append(data['adnames'][ç])
            ç +=1
    elif len(data['allad'][0]) == 96:
        time = np.arange(0,len(data['allad'][0][64])/data['adfreq'][0][0],1/data['adfreq'][0][0]) 
        values = np.zeros((len(data['allad'][0][64]),32))
        ç = 64
        for j in range(32):
            for r in range(len(data['allad'][0][64])):
                values[r][j] = data['allad'][0][ç][r]
            column_name.append(data['adnames'][ç])
            ç +=1

    df = pd.DataFrame(values,columns=column_name,index= None) 
    df.insert(0, "Time", time, True)

    return df  

def PlotInteraction(interactions, **kwargs):
  saveName= kwargs.get('saveName')
  start= kwargs.get('start')
  end= kwargs.get('end')
  figureTitle = kwargs.get('figureTitle')
  fps = kwargs.get('fps')
  ax = kwargs.get('ax')
  aspect = kwargs.get('aspect')
  if type(aspect) == type(None):
    aspect = 'equal'
  if type(fps) == type(None):
    fps = 30
  hSize = kwargs.get('hSize')
  if type(hSize) == type(None):
    hSize = 2
  wSize = kwargs.get('wSize')
  if type(wSize) == type(None):
    wSize = 8
  fontsize = kwargs.get('fontsize')
  if type(fontsize) == type(None):
    fontsize = 15
  figformat = kwargs.get('figformat')
  if type(figformat) == type(None):
    figformat = '.eps'
    res = kwargs.get('res')
  if type(res) == type(None):
    res = 80 
  barH = kwargs.get('barH')
  if type(barH) == type(None):
    barH = .5

  if type(start) == type(None):
      init = 0
      finish = interactions.end.iloc[-1]
  else:
      init = int(start)
      finish = int(end) 

  times = []
  starts = []
  for i in range (int(interactions.obj.max())+1):
      times.append((interactions.end.loc[(interactions.obj == i) & (interactions.start >= init) & (interactions.start <= finish)])-(interactions.start.loc[(interactions.obj == i) & (interactions.start >= init) & (interactions.start <= finish)]).values)
      starts.append((interactions.start.loc[(interactions.obj == i) & (interactions.start >= init) & (interactions.start <= finish)]).values)

  barHeight = barH

  if type(ax) == type(None):
    plt.figure(figsize=(wSize,hSize))
    for i in range (1,int(interactions.obj.max())+1):
      plt.barh(0,times[i], left=starts[i], height = barHeight, label = "Field "+str(i))

    plt.title(figureTitle,fontsize=fontsize)
    plt.legend(ncol=int(interactions.obj.max()))
    plt.xlim(init, finish)
    plt.yticks([])
    plt.xticks(fontsize = fontsize*0.8)
    plt.xlabel("Time (s)",fontsize=fontsize)
    plt.ylim([-barHeight,barHeight])

    if type(saveName) != type(None):
        plt.savefig(saveName+figformat)

    plt.show()

  else:
    for i in range (1,int(interactions.obj.max())+1):
      ax.barh(0,times[i], left=starts[i], height = barHeight, label = "Field "+str(i))
    ax.set_title(figureTitle, fontsize=fontsize)
    if aspect == type(None):
      ax.set_aspect(aspect)
    ax.set_xlim([init, finish])
    ax.set_yticklabels([])
    ax.get_yaxis().set_visible(False)
    ax.tick_params(axis='x', labelsize=fontsize*.8)
    ax.tick_params(axis='y', labelsize=fontsize*.8)
    ax.legend(ncol=int(interactions.obj.max()),fontsize=fontsize*.8)
    ax.set_xlabel('Time (s)',fontsize=fontsize)

    ax.set_ylim([-barHeight,barHeight])

def Blackrock(data_path, freq): 
    from neo.io import BlackrockIO
    
    reader = BlackrockIO(data_path)
    seg = reader.read_segment()

    column_name = []
    time = np.arange(0,len(seg.analogsignals[0])/freq,1/freq)
    values = np.zeros((len(seg.analogsignals[0]),len(seg.analogsignals[0][0])))

    for i in range(len(seg.analogsignals[0][0])):
        for ç in range(len(seg.analogsignals[0])):
            values[ç][i] = float(seg.analogsignals[0][ç][i])

    channels = []
    for i in range(len(seg.analogsignals[0][0])):
        channels.append('Channel ' + str(i+1))

    df = pd.DataFrame(values,columns=channels,index= None) 
    df.insert(0, "Time", time, True)

    return df

def SpacialNeuralActivity(neural_data, unit):
    neural_data = neural_data.loc[ neural_data['x'] > 100, : ]

    xmin, xmax = neural_data['x'].min(), neural_data['x'].max()
    ymin, ymax = neural_data['y'].min(), neural_data['y'].max()

    xsteps = np.linspace(xmin, xmax, num=100)
    ysteps = np.linspace(ymin, ymax, num=100)

    heatmap = np.zeros( (xsteps.shape[0], ysteps.shape[0]) )

    for x in range(xsteps.shape[0]-1):
        for y in range(ysteps.shape[0]-1):
            df_tmp = neural_data.loc[ (neural_data['x'] >= xsteps[x]) & (neural_data['x'] < xsteps[x+1]) &
                                (neural_data['y'] >= ysteps[y]) & (neural_data['y'] < ysteps[y+1]), : ]
            heatmap[x, y] = df_tmp[unit].sum()
            
    return heatmap

def IntervalBehaviors(cluster_labels, fps=30 , filter = 10, correction = 0):
    cluster_num = set(cluster_labels)
    intervals = {}
    for ç in cluster_num:

        index = np.where(cluster_labels==ç)

        dicts = {}
        dicts2 = {}
        count = 0
        init = []
        end = []

        for i in range(len(index[0])-1):
                if index[0][i+1] - index[0][i] <=filter:
                    count +=1    
                if index[0][i+1] - index[0][i] >filter:
                    dicts[i] = index[0][i] + correction
                    dicts2[i] = index[0][i] - count + correction
                    count = 0

        for i in range(len(list(zip(dicts2.values(),dicts.values())))):
            if list(zip(dicts2.values(),dicts.values()))[i][0] != list(zip(dicts2.values(),dicts.values()))[i][1]:
                init.append(int(list(zip(dicts2.values(),dicts.values()))[i][0]/fps))
                end.append(int(list(zip(dicts2.values(),dicts.values()))[i][1]/fps))

        intervals[ç] = (init,end)
    
    return intervals
    