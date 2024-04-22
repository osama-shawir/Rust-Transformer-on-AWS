use lambda_http::{run, service_fn, tracing, Body, Error, Request, RequestExt, Response};
use std::{convert::Infallible, io::Write, path::PathBuf};
/// This is the main body for the function.
/// Write your code inside it.
/// There are some code example in the following URLs:
/// - https://github.com/awslabs/aws-lambda-rust-runtime/tree/main/examples
fn infer(prompt: String) -> Result<String, Box<dyn std::error::Error>> {
    let tokenizer_source = llm::TokenizerSource::Embedded;
    let model_architecture = llm::ModelArchitecture::GptNeoX;
    // for local run
    // let model_path = PathBuf::from("src/pythia-1b-q4_0-ggjt.bin");
    // for deployment
    let model_path = PathBuf::from("/new-lambda-project/pythia-1b-q4_0-ggjt.bin");

    let prompt = prompt.to_string();
    let model = llm::load_dynamic(
        Some(model_architecture),
        &model_path,
        tokenizer_source,
        Default::default(),
        llm::load_progress_callback_stdout,
    )?;

    let mut session = model.start_session(Default::default());
    let mut generated_tokens = String::new();

    let res = session.infer::<Infallible>(
        model.as_ref(),
        &mut rand::thread_rng(),
        &llm::InferenceRequest {
            prompt: (&prompt).into(),
            parameters: &llm::InferenceParameters::default(),
            play_back_previous_tokens: false,
            // specify token to use
            maximum_token_count: Some(20),
        },
        // OutputRequest
        &mut Default::default(),
        |r| match r {
            llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
                print!("{t}");
                std::io::stdout().flush().unwrap();
                generated_tokens.push_str(&t);
                Ok(llm::InferenceFeedback::Continue)
            }
            _ => Ok(llm::InferenceFeedback::Continue),
        },
    );

    // Return statement
    match res {
        Ok(_) => Ok(generated_tokens),
        Err(err) => Err(Box::new(err)),
    }
}

async fn function_handler(event: Request) -> Result<Response<Body>, Error> {
    // Extract some useful information from the request
    let query = event
        .query_string_parameters_ref()
        .and_then(|params| params.first("query"))
        .unwrap_or("I stayed up all night to");
    let message = match infer(query.to_string()) {
        Ok(inference_result) => {
            format!("{}", inference_result)
        }
        Err(err) => {
            format!("Error in inference: {:?}", err)
        }
    };
    println!("Model generated response");
    println!("{:?}", message);

    let resp = Response::builder()
        .status(200)
        .header("content-type", "text/html")
        .body(message.into())
        .map_err(Box::new)?;
    Ok(resp)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing::init_default_subscriber();

    run(service_fn(function_handler)).await
}


