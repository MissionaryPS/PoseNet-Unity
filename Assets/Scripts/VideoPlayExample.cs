using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;
using UnityEngine.UI;
using TensorFlow;

public class VideoPlayExample : MonoBehaviour {
	public bool IsWebcam = true;

	WebCamTexture webcamTexture;
	RenderTexture videoTexture;

	int videoWidth;
	int videoHeight;
	int FPS = 30;

	static int detectWidth = 256;
	static int detectHeight = 256;

    [SerializeField]
	GLRenderer gl;
    [SerializeField]
    GameObject vrmModel;

    PoseNet posenet = new PoseNet();
	PoseNet.Pose[] poses;
	float minPoseConfidence = 0.3f;
	float minPartConfidence = 0.0f;
	TFSession session;
	TFGraph graph;
	bool isPosing;

	Dictionary<string, GameObject> dst = new Dictionary<string, GameObject>();

    void Start () {
        webcamTexture = new WebCamTexture(WebCamTexture.devices[0].name);

		Application.runInBackground = true;

		RectTransform rectTransform = GetComponent<RectTransform>();
        Renderer renderer = GetComponent<Renderer>();

		videoWidth = (int)rectTransform.rect.width;
		videoHeight = (int)rectTransform.rect.height;

		if(IsWebcam){
			WebCamDevice[] devices = WebCamTexture.devices;
			webcamTexture = new WebCamTexture(devices[0].name, videoWidth, videoHeight, FPS);
			renderer.material.mainTexture = webcamTexture;
			webcamTexture.Play();

			texture = new Texture2D(webcamTexture.width, webcamTexture.height);

		}else{
			VideoPlayer videoPlayer = GetComponent<VideoPlayer>();
			videoTexture = new RenderTexture(videoWidth, videoHeight, 24);
			videoPlayer.targetTexture = videoTexture;
			renderer.material.mainTexture = videoTexture;
			videoPlayer.Play();

			texture = new Texture2D(videoTexture.width, videoTexture.height);
		}
		scaleTexture = new Texture2D(detectWidth, detectHeight, TextureFormat.ARGB32, true);
		scaleRenderTexture = new RenderTexture(detectWidth, detectHeight, 32);

		TextAsset graphModel = Resources.Load("frozen_model") as TextAsset;
		graph = new TFGraph();
		graph.Import(graphModel.bytes);
		session = new TFSession(graph);

        //ここからtakasaka
        Debug.Log(vrmModel.name);
        Dictionary<string, GameObject> tmp = new Dictionary<string, GameObject>();
		GetJoints(vrmModel, ref tmp);
        
        dst["hips"]       = tmp["J_Bip_C_Hips"];
		dst["spine"]      = tmp["J_Bip_C_Spine"];
		dst["chest"]      = tmp["J_Bip_C_Chest"];
		dst["upperChest"] = tmp["J_Bip_C_UpperChest"];
		dst["neck"]       = tmp["J_Bip_C_Neck"];
		dst["head"]       = tmp["J_Bip_C_Head"];
		dst["upperArmL"]  = tmp["J_Bip_L_UpperArm"];	dst["upperArmR"]  = tmp["J_Bip_R_UpperArm"];
		dst["lowerArmL"]  = tmp["J_Bip_L_LowerArm"];	dst["lowerArmR"]  = tmp["J_Bip_R_LowerArm"];
		dst["handL"]      = tmp["J_Bip_L_Hand"];		dst["handR"]      = tmp["J_Bip_R_Hand"];
		dst["upperLegL"]  = tmp["J_Bip_L_UpperLeg"];	dst["upperLegR"]  = tmp["J_Bip_R_UpperLeg"];
		dst["lowerLegL"]  = tmp["J_Bip_L_LowerLeg"];	dst["lowerLegR"]  = tmp["J_Bip_R_LowerLeg"];
		dst["footL"]      = tmp["J_Bip_L_Foot"];		dst["footR"]      = tmp["J_Bip_R_Foot"];
        //ここまでtakasaka
	}

	Texture2D texture;

	void Update () {
        
		if(IsWebcam){
			var color32 = webcamTexture.GetPixels32();
			texture.SetPixels32(color32);
			texture.Apply();

		}else{
			RenderTexture.active = videoTexture;
			texture.ReadPixels(new Rect(0, 0, videoTexture.width, videoTexture.height), 0, 0);
			texture.Apply();
			RenderTexture.active = null;
		}

		if (isPosing) return;
		isPosing = true;
		StartCoroutine("PoseUpdate", texture);
        
	}



	Dictionary<string, PoseNet.Keypoint> src = new Dictionary<string, PoseNet.Keypoint>();
	Dictionary<string, Vector3> joint = new Dictionary<string, Vector3>();
	Dictionary<int, Dictionary<string, Vector3>> jointsAvg = new Dictionary<int, Dictionary<string, Vector3>>();
	int jointsCount = 10, jointsIndex = 0;

    IEnumerator PoseUpdate(Texture2D texture)
    {
		if(texture.width != detectWidth || texture.height != detectHeight){
	        texture = scaled(texture, detectWidth, detectHeight);
        }
        var tensor = TransformInput(texture.GetPixels32());

        var runner = session.GetRunner();
        runner.AddInput(graph["image"][0], tensor);
        runner.Fetch(
            graph["heatmap"][0],
            graph["offset_2"][0],
            graph["displacement_fwd_2"][0],
            graph["displacement_bwd_2"][0]
        );

        var result = runner.Run();
        var heatmap = (float[,,,])result[0].GetValue(jagged: false);
        var offsets = (float[,,,])result[1].GetValue(jagged: false);
        var displacementsFwd = (float[,,,])result[2].GetValue(jagged: false);
        var displacementsBwd = (float[,,,])result[3].GetValue(jagged: false);

       // Debug.Log(PoseNet.mean(heatmap));

        poses = posenet.DecodeMultiplePoses(
            heatmap, offsets,
            displacementsFwd,
            displacementsBwd,
            outputStride: 16, maxPoseDetections: 1,
            scoreThreshold: 0.5f, nmsRadius: 20);

        isPosing = false;

        //ここからtakasakaのコード
		if(poses.Length > 0 && poses[0].score >= minPoseConfidence){
			var pose = poses[0];

			src.Clear();
			src["nose"]      = pose.keypoints[0];
			src["eyeL"]      = pose.keypoints[1]; src["eyeR"]       = pose.keypoints[2];
			src["earL"]      = pose.keypoints[3]; src["earR"]       = pose.keypoints[4];
			src["upperArmL"] = pose.keypoints[5]; src["upperArmR"]  = pose.keypoints[6];
			src["lowerArmL"] = pose.keypoints[7]; src["lowerArmR"]  = pose.keypoints[8];
			src["handL"]     = pose.keypoints[9]; src["handR"]      = pose.keypoints[10];
			src["upperLegL"] = pose.keypoints[11]; src["upperLegR"] = pose.keypoints[12];
			src["lowerLegL"] = pose.keypoints[13]; src["lowerLegR"] = pose.keypoints[14];
			src["footL"]     = pose.keypoints[15]; src["footR"]     = pose.keypoints[16];

			joint.Clear();
			foreach(KeyValuePair<string, PoseNet.Keypoint> pair in src){
				if(pair.Value.score < minPartConfidence){ continue; }

				//PoseNetの機能強化あるいは他の姿勢推定ライブラリに切り替えられるようVector3で作っておく
				joint[pair.Key]  = new Vector3(pair.Value.position.x, videoHeight - pair.Value.position.y, 0);
			}

			//前フレームのジョイント位置と平均を取る
			if(jointsAvg.Count == 0){
				for(int i = 0; i < jointsCount; i++){
					jointsAvg[i] = new Dictionary<string, Vector3>(joint);
				}
			}
			jointsAvg[jointsIndex++ % jointsCount] = new Dictionary<string, Vector3>(joint);

			foreach(string key in src.Keys){
				if(!joint.ContainsKey(key)){ continue; }
				joint[key] = Vector3.zero;
				
				int hit = 0;
				for(int i = 0; i < jointsCount; i++){
					if(!jointsAvg[i].ContainsKey(key)){ continue; }

					joint[key] += jointsAvg[i][key];
					++hit;
				}
				if(hit > 0){ joint[key] /= hit; }
			}

			//左腕
			if(joint.ContainsKey("upperArmL")){
				if(joint.ContainsKey("lowerArmL")){
					if(joint.ContainsKey("upperArmR")){ 
						UpdateJoint(joint, "upperArmR", "upperArmL", "lowerArmL", dst["upperArmL"]); 
						AdjJoint(0, -40, 0, dst["upperArmL"]); //ダミー
					}

					if(joint.ContainsKey("handL")){ 
						UpdateJoint(joint, "upperArmL", "lowerArmL", "handL", dst["lowerArmL"]); 

						//常に手のひらを向けるよう補正
						var armLow2Hand = (joint["handL"] - joint["lowerArmL"]);
						armLow2Hand.Normalize();
						var angleX = Rad2Deg(armLow2Hand.y) + 90;
						var angleY = Mathf.Min(0, Rad2Deg(armLow2Hand.x));
						AdjJoint(angleX, angleY, 0, dst["lowerArmL"]);

						dst["handL"].transform.localRotation = new Quaternion();
						AdjJoint(0, 0, -20, dst["handL"]); //ダミー
					}
				}
			}

			//右腕
			if(joint.ContainsKey("upperArmR")){
				if(joint.ContainsKey("lowerArmR")){
					if(joint.ContainsKey("upperArmL")){ 
						UpdateJoint(joint, "upperArmL", "upperArmR", "lowerArmR", dst["upperArmR"]); 
						AdjJoint(0, 40, 0, dst["upperArmR"]); //ダミー
					}
					if(joint.ContainsKey("handR")){ 
						UpdateJoint(joint, "upperArmR", "lowerArmR", "handR", dst["lowerArmR"]); 

						//常に手のひらを向けるよう補正
						var armLow2Hand = joint["handR"]- joint["lowerArmR"];
						armLow2Hand.Normalize();
						var angleX = Rad2Deg(armLow2Hand.y) + 90;
						var angleY = Mathf.Max(0, Rad2Deg(armLow2Hand.x));
						AdjJoint(angleX, angleY, 0, dst["lowerArmR"]);

						dst["handR"].transform.localRotation = new Quaternion();
						AdjJoint(0, 0, 20, dst["handR"]); //ダミー
					}
				}
			}
			//胸
			if(joint.ContainsKey("upperArmL") && joint.ContainsKey("upperArmR")){
				joint["upperArmLL"] = joint["upperArmL"] + vecX;
				UpdateJoint(joint, "upperArmLL", "upperArmL", "upperArmR", dst["upperChest"], -20, 20);
			}
			//腰
			if(joint.ContainsKey("upperLegL") && joint.ContainsKey("upperLegR")){
				//基準点が無いので左肩の左水平方向に仮のジョイントを作る
				joint["upperLegLL"] = joint["upperLegL"] + vecX;
				UpdateJoint(joint, "upperLegLL", "upperLegL", "upperLegR", dst["spine"], -10, 10);

				float addX = -3.0f;
				float mulX = 10.0f;

				var pos = joint["upperLegL"] + joint["upperLegR"];
				pos /= 2;
				var x = -(pos.x - (videoWidth / 2)) / videoWidth;
				
				Vector3 tmp = dst["hips"].transform.position;
				dst["hips"].transform.position = new Vector3(x * mulX + addX, tmp.y, tmp.z);

				//AdjJoint(-20, 0, 0, dst["spine"]); //ダミー
			}
			//左脚
			if(joint.ContainsKey("upperLegL")){
				if(joint.ContainsKey("lowerLegL")){
					if(joint.ContainsKey("upperLegR")){
						//基準点が無いので左脚付け根の上方向に仮のジョイントを作る
						joint["upperLegLUp"] = joint["upperLegR"] - joint["upperLegL"];
						joint["upperLegLUp"].Normalize();
						joint["upperLegLUp"] = Quaternion.AngleAxis(Rad2Deg(-halfPi), vecZ) * joint["upperLegLUp"];
						joint["upperLegLUp"] += joint["upperLegL"];
						UpdateJoint(joint, "upperLegLUp", "upperLegL", "lowerLegL", dst["upperLegL"], -20, 20); 
					}
					if(joint.ContainsKey("footL")){ 
						UpdateJoint(joint, "upperLegL", "lowerLegL", "footL", dst["lowerLegL"], -20, 20); 

						//基準点が無いので左足首の下垂直方向に仮のジョイントを作る
						joint["footLDown"] = joint["footL"] - vecY;
						UpdateJoint(joint, "lowerLegL", "footL", "footLDown", dst["footL"]); 
					}else{
						//基準点が無いので左膝の下垂直方向に仮のジョイントを作る
						joint["lowerLegLDown"] = joint["lowerLegL"] - vecY;
						UpdateJoint(joint, "upperLegL", "lowerLegL", "lowerLegLDown", dst["lowerLegL"]); 
						UpdateJoint(joint, "lowerLegL", "lowerLegLDown", "lowerLegLDown", dst["footL"]); 
					}
					AdjJoint(0, 10, 0, dst["footL"]); //ダミー
				}
			}
			//右脚
			if(joint.ContainsKey("upperLegR")){
				if(joint.ContainsKey("lowerLegR")){
					if(joint.ContainsKey("upperLegL")){
						//基準点が無いので右脚付け根の上方向に仮のジョイントを作る
						joint["upperLegRUp"] = joint["upperLegL"] - joint["upperLegR"];
						joint["upperLegRUp"].Normalize();
						joint["upperLegRUp"] = Quaternion.AngleAxis(Rad2Deg(halfPi), vecZ) * joint["upperLegRUp"];
						joint["upperLegRUp"] += joint["upperLegR"];
						UpdateJoint(joint, "upperLegRUp", "upperLegR", "lowerLegR", dst["upperLegR"], -20, 20); 
					}
					if(joint.ContainsKey("footR")){ 
						UpdateJoint(joint, "upperLegR", "lowerLegR", "footR", dst["lowerLegR"], -20, 20); 

						//基準点が無いので右足首の下垂直方向に仮のジョイントを作る
						joint["footRDown"] = joint["footR"] - vecY;
						UpdateJoint(joint, "lowerLegR", "footR", "footRDown", dst["footR"]); 
					}else{
						//基準点が無いので右膝の下垂直方向に仮のジョイントを作る
						joint["lowerLegRDown"] = joint["lowerLegR"] - vecY;
						UpdateJoint(joint, "upperLegR", "lowerLegR", "lowerLegRDown", dst["lowerLegR"]); 
						UpdateJoint(joint, "lowerLegR", "lowerLegRDown", "lowerLegRDown", dst["footR"]); 
					}
					AdjJoint(0, -10, 0, dst["footR"]); //ダミー
				}
			}
        }
        //ここまでtakasakaのコード

        yield return null;
    }

	static Vector3 vecX = new Vector3(1, 0, 0);
	static Vector3 vecY = new Vector3(0, 1, 0);
	static Vector3 vecZ = new Vector3(0, 0, 1);
	static float halfPi = Mathf.PI / 2.0f;

	static float Deg2Rad(float deg){ return deg * Mathf.PI / 180.0f; }
	static float Rad2Deg(float rad){ return rad * 180.0f / Mathf.PI; }

	//srcのroot→nodeとnode→leafの角度でdstを回転
	static void UpdateJoint(Dictionary<string, Vector3> src, string root, string node, string leaf, 
							GameObject dst, float min = -360.0f, float max = 360.0f){

		var from = (src[leaf] - src[node]);
		from.Normalize();
		var to = (src[node] - src[root]);
		to.Normalize();
		var axis = Vector3.Cross(from, to);
		axis.Normalize();
		var angle = Mathf.Acos(Vector3.Dot(from, to));
		angle = Mathf.Max(Deg2Rad(min), Mathf.Min(Deg2Rad(max), angle));
		var quat = Quaternion.AngleAxis(Rad2Deg(angle), axis);
		dst.transform.localRotation = quat;
	}
	
	//回転角度の調整
	static void AdjJoint(float x, float y, float z, GameObject dst){
		dst.transform.localRotation *= Quaternion.Euler(-x, -y, -z);
	}

    public void OnRenderObject()
    {
        //Debug.Log(poses);
        gl.DrawResults(poses, minPoseConfidence);
    }

    static float[] floatValues = new float[detectWidth * detectHeight * 3];
    static TFShape shape = new TFShape(1, detectWidth, detectHeight, 3);
    public static TFTensor TransformInput(Color32[] pic)
    {
		float tmp = 2.0f / 255.0f;
		int p = 0;
		for (int h = detectHeight -1; h >= 0; --h){
			for(int w = 0; w < detectWidth; ++w){
				var color = pic[h * detectWidth + w];
				floatValues[p * 3 + 0] = color.r * tmp - 1.0f;
				floatValues[p * 3 + 1] = color.g * tmp - 1.0f;
				floatValues[p * 3 + 2] = color.b * tmp - 1.0f;
				++p;
			}
		}

        return TFTensor.FromBuffer(shape, floatValues, 0, floatValues.Length);
    }

    static Rect scaleRect = new Rect(0, 0, detectWidth, detectHeight);
    static Texture2D scaleTexture;
    public static Texture2D scaled(Texture2D src, int width, int height, FilterMode mode = FilterMode.Trilinear)
    {
        _gpu_scale(src, mode);

        //Get rendered data back to a new texture
        //scaleTexture.Resize(detectWidth, detectHeight);
        scaleTexture.ReadPixels(scaleRect, 0, 0, true);
        return scaleTexture;
    }

    //Using RTT for best quality and performance. Thanks, Unity 5
    static RenderTexture scaleRenderTexture;
    static void _gpu_scale(Texture2D src, FilterMode fmode)
    {
        //We need the source texture in VRAM because we render with it
        src.filterMode = fmode;
        src.Apply(true);

        //Set the RTT in order to render to it
        Graphics.SetRenderTarget(scaleRenderTexture);

        //Setup 2D matrix in range 0..1, so nobody needs to care about sized
        GL.LoadPixelMatrix(0, 1, 1, 0);

        //Then clear & draw the texture to fill the entire RTT.
        GL.Clear(true, true, new Color(0, 0, 0, 0));
        Graphics.DrawTexture(new Rect(0, 0, 1, 1), src);
    }

	public static void GetJoints(GameObject obj, ref Dictionary<string, GameObject> dst)
	{
		Transform children = obj.GetComponentInChildren<Transform> ();
		if (children.childCount == 0) { return; }

		foreach (Transform ob in children) {
			dst[ob.name] = ob.gameObject;
			GetJoints(ob.gameObject, ref dst);
		}
	}
}
