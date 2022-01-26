const RootStyle = styled(Page)(({ theme }) => ({
    paddingTop: theme.spacing(8),
    marginTop: theme.spacing(8),
    marginBottom: theme.spacing(8),
    [theme.breakpoints.up("md")]: {
      paddingTop: theme.spacing(11),
    },
  }));
  
  const useStyles = makeStyles((theme) => ({
    buttonDesktop: {
      height: 60,
      width: "100%",
      justifyContent: "flex-start",
      borderRadius: "0 10px 10px 0",
    },
    buttonMobile: {
      height: 60,
      width: "100%",
      justifyContent: "flex-start",
      borderRadius: "10px",
    },
  }));
  
  const Item = styled(Paper)(({ theme }) => ({
    ...theme.typography.body2,
    padding: theme.spacing(1),
    textAlign: 'center',
    color: theme.palette.text.secondary,
  }));
  
  const data = {
    item: {
      jobTitle: "Senior OMS Developer",
      location: "Seatle, WA (Remote)",
      jobType : "Contract",
      salary: '$80 - $90 per hour',
      datePosted: moment.now(),
    },
  };
  
  const ContentStyle = styled("div")(({ theme }) => ({
    width: "100%",
    textAlign: "center",
    marginBottom: theme.spacing(10),
    [theme.breakpoints.up("md")]: {
      textAlign: "left",
      marginBottom: 0,
    },
  }));
  const ContentItem = (props) => (
   // <MotionInView variants={varFadeInUp}>
      <Stack
        direction="row"
        sx={{
          mb: 1,
        }}
      >
        <Typography>
          <span style={{ fontWeight: "bold" }}>{`${props.title} -`}</span>
        </Typography>
        <Typography color="secondary">
          <span style={{ fontWeight: "bold" }}>&nbsp;{props.value}</span>
        </Typography>
      </Stack>
   // </MotionInView>
  );
  
  function PostItem1(props) {
    const {
      jobTitle,
      jobtitle,
      location,
      jobType,
      salary,
      datePosted,
      applyCurrentJob
    } = props;
  
      const { user, isAuthenticated } = useAuth();
      console.log('props inside postItem1 within same file')
   //   console.log(props)
    return (
      <RootStyle>
        {/* <Container maxWidth="lg"> */}
        <Paper
          sx={{
            py: 3,
            px: 4,
            width: "100%",
            zIndex: (theme) => theme.zIndex.modal,
            boxShadow: (theme) => theme.customShadows.z20,
            bgcolor:"text.disabled",
          }}
        >
           <div >
                  <Typography variant="h5" sx={{paddingLeft:3.9, mb: 2, color:"white" }}>
                    <span
                      style={{ fontWeight: "bold" }}
                    >{`${props['Job Title']}`}</span>
                  </Typography>
            </div>
          <Grid
            container
            // spacing={5}
            justifyContent="flex-start"
          >
            <Grid
              item
              xs={12}
              md={12}
              sx={{ alignitems:"center", px: 4, color:"white" }}
            >
              <ContentStyle >
               <ContentItem title="Location📍"  value={props['Location']} /><hr/>
               <Typography 
               sx={{alignItems:"center", paddingTop:2, paddingBottom:2, textAlign:"center"}}
               title="Job Details"
               >
                 Job Details
               </Typography><hr/>
              
              <br/>{props['Company'] && <ContentItem title="Company" value={props['Company']} />}<br/>
               {props['Job Description']&& <ContentItem title="Job Description" />} 
               {props['Job Description']}<br/><br/>
               <ContentItem title="Job Link" value={props['Job Link']} />
              <br/><ContentItem title="Date Posted" value={props['Processed_Date']} />
              <div>
        <h1>Job Details!</h1>
             {jobtitle.map(jobTitle => 
             {console.log(jobTitle); 
              return <div>{jobTitle.job_role_title}<br/><br/><br/>{jobTitle.job_description_text}<br/><br/><br/><br/>{jobTitle.Apply_Link}<br/><br/><br/><br/>{jobTitle.Processed_Date}</div>}            
              )}
      </div>
              </ContentStyle>
              
            </Grid>
  
            </Grid>
  
            <Button
                      variant="contained"
                      color="secondary"
                      size="large"
                      type="submit"
       //               className={classes.buttonDesktop}
       //                startIcon={<SearchIcon />}
                       to={
                          isAuthenticated
                          ? `${PATH_PAGE.jobs}/${props['_id']}`
                          : PATH_AUTH[`candidateLogin`]
                      }
                      //onClick={()=>{applyCurrentJob(props['_id'])}}
                      component={RouterLink}
                    > 
                      Apply
              </Button>
  
        </Paper>
        {/* </Container> */}
      </RootStyle>
    );
  }
  
  export default function PostSearch() {
    const classes = useStyles();
    const routeParams = useParams({});
    const { user } = useAuth();
    const [jobData, SetJobData] = useState([]);
    const [selectedJobID, SetSelectedJobID]= useState('')
    const [showResults, setShowResults] = useState(false)
    const [showApplicationForm , setShowApplicationForm] = useState(false)
    const [jobtitle, setJobtitle] = useState([]);
  
    console.log('inside post search route params ')
    console.log(routeParams)
  
  
  {/* useEffect( ()=>{
      setShowResults(false)
        axios.post('https://us-central1-rekommenderbackend.cloudfunctions.net/irekommend-ui-candidate-job-search',
            { search_string: routeParams.job_title + ' ' + routeParams.location,
                   // role_desc:'routeParams.roleDescription',
                  //  role_location: 'NA',   //routeParams.location
                  //  education_level:'NA',
                    // years_of_exp_required: 'NA',
                    // resume_last_updated_range:'NA',
                  //  user_email: user.email 
                }
      ).then((data)=>{
          console.log('loadJobDataFromServer')
          console.log(data)    
          SetJobData(data)
          console.log(SetJobData)
          setShowResults(true)
  
          })
    } , []
  )*/}
  
  useEffect( ()=>{
    setShowResults(false)
    axios.post('https://irekommend-ml-utility.ue.r.appspot.com/search-jobs?search_string=%22Microservices%20and%20AWS%22',
        { search_string: routeParams.job_title + ' ' + routeParams.location,
                 // role_desc:'routeParams.roleDescription',
                //  role_location: 'NA',   //routeParams.location
                //  education_level:'NA',
                  // years_of_exp_required: 'NA',
                  // resume_last_updated_range:'NA',
                //  user_email: user.email 
        }
    ).then((data)=>{
        console.log('loadJobDataFromServer')
        console.log(data)    
        SetJobData(data)
        console.log(SetJobData)
        setShowResults(true)
  
        })
  } , []
  )
  
  
  {/*useEffect(() => {
    fetch("https://irekommend-ml-utility.ue.r.appspot.com/search-jobs?search_string=%22Microservices%20and%20AWS%22")
    .then((response) => response.json())
    .then((json) => {
      console.log(json)
      setJobtitle(json);
    });
  }, []);*/}
  
    const applyCurrentJob = (key) =>{
      console.log('applying for this job ID ' +  key )
      console.log('applying for this candidate ' +  user.email )
      SetSelectedJobID(key)
      setShowApplicationForm(true)
      
    }
    return (
      <RootStyle title="iRekommend | Jobs">
        <Container maxWidth="lg">
          <Grid item xs={12} sx={{ my: 4 }}>
            <Grid
              container
              direction="row"
              justifyContent="center"
              alignItems="center"
              spacing={0.5}
            >
              
              {/* <Grid item xs={10}>
                <KeywordTextFieldDesktop
                  label=" Enter Job titles, keywords etc. or copy paste your job description here"
                  variant="filled"
                  id="keyword-input"
                />
              </Grid>
              <Grid item xs={2} style={{ height: "100%" }}>
                <Button
                  variant="contained"
                  color="primary"
                  size="large"
                  className={classes.buttonDesktop}
                  startIcon={<SearchIcon />}
                >
                  Search
                </Button>
              </Grid> */}
            </Grid>
          </Grid>
          
          <Grid container spacing={2}>
            {/* {routeParams.from === "Employer" && (
              <Grid item xs={12} md={4} sx={{ pr: 1 }}>
                <SearchPanel {...data.item} />
              </Grid>
            )} */}
            <Grid
              item
              xs={6}
              md={routeParams.from === "Employer" ? 8 : 12}
              sx={{ pl: routeParams.from === "Employer" ? 1 : 0 }}
            >
                  {/* <Grid item xs={12} sx={{ my: 1 }}>
                    <TextField
                      sx={{ width: "100%" }}
                    // label="500 jobs found"
                      disabled={true}
                    />
                  </Grid> */}
                <Item> { showResults && jobData.data.results.map((job, key)=>{
                      return  <PostItem1 {...job} key = {key} applyCurrentJob={applyCurrentJob}/>
                  })}   </Item>             
                  {/* <PostItem1 {...data.item} />
                  <PostItem1 {...data.item} />
                  <PostItem1 {...data.item} /> */}
            </Grid>
            <Grid
              item
              xs={4}
            >
                 <Item> {showApplicationForm && <CandidateRegisterForm selectedJobID={selectedJobID}/>}</Item>  
  
            </Grid>
            
          </Grid>
        </Container>
        <div>
        <h1>Job Details!</h1>
             {jobtitle.map(jobTitle => 
             {console.log(jobTitle); 
              return <div>{jobTitle.job_role_title}<br/><br/><br/>{jobTitle.job_description_text}<br/><br/><br/><br/>{jobTitle.Apply_Link}<br/><br/><br/><br/>{jobTitle.Processed_Date}</div>}            
              )}
      </div>
      </RootStyle>
    );
  }
  