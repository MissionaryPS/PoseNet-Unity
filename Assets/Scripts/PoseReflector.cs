using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PoseReflector : MonoBehaviour
{
    [SerializeField]
    GameObject humanoid;    //unity humanoid準拠であればなんでもよい

    Animator animator = null;

    // Start is called before the first frame update
    void Start()
    {
        animator = humanoid.GetComponent<Animator>();
    }

    void PoseNet2VRM(PoseNet.Pose[] poses)
    {

        //shoulder

        //hips

        //right upper arm
        //right under arm
        //left upper arm
        //left under arm

        //right upper leg
        //right under leg
        //left upper leg
        //left under leg



    }

}
