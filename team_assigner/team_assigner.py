from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self,image):
        image_2d = image.reshape(-1,3)                                # Reshape the image to 2D array

        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)      # Preform K-means with 2 clusters, different init method for speed
        kmeans.fit(image_2d)

        return kmeans
    
    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        kmeans = self.get_clustering_model(top_half_image)                                      # Get Clustering model

        labels = kmeans.labels_                                                                 # Get the cluster labels forr each pixel

        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])       # Reshape the labels to the image shape

        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster                                                 # Get the player cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color
    
    def assign_team_color(self,frame,player_detections):

        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame,bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:                               # if player_id exists in dict...
            return self.player_team_dict[player_id]                          # ...return the team
        
        player_color = self.get_player_color(frame,player_bbox)              # run kmeans to get player color

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]         # get team id by running self.kmeans
        team_id += 1

        # if player_id == 126:                                               # So that Onana is red
        #     team_id = 2
        
        # if player_id == 1052:                                               # So that Ortega is blue
        #     team_id = 1
        
        # if player_id == 17:                                                   # So that Mainoo is red
        #     team_id = 2
        
        self.player_team_dict[player_id] = team_id                           # save team id in dict

        return team_id