/// Official Task Prompts for EmbeddingGemma 2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Task<'a> {
    #[default]
    SearchQuery,
    Document { title: Option<&'a str> },
    CodeRetrieval,
    Classification,
    Clustering,
    QuestionAnswering,
    FactChecking,
    SentenceSimilarity,
    Custom(&'a str),
    Raw,
}

impl<'a> Task<'a> {
    /// Returns the official prompt prefix string for this task, if any.
    pub fn prompt_prefix(&self) -> Option<String> {
        match self {
            Task::SearchQuery => Some("task: search result | query: ".to_string()),
            Task::Document { title } => match title {
                Some(t) => Some(format!("title: {} | text: ", t)),
                None => Some("title: none | text: ".to_string()),
            },
            Task::CodeRetrieval => Some("task: code retrieval | query: ".to_string()),
            Task::Classification => Some("task: classification | query: ".to_string()),
            Task::Clustering => Some("task: clustering | query: ".to_string()),
            Task::QuestionAnswering => Some("task: question answering | query: ".to_string()),
            Task::FactChecking => Some("task: fact checking | query: ".to_string()),
            Task::SentenceSimilarity => Some("task: sentence similarity | query: ".to_string()),
            Task::Custom(s) => {
                if s.contains(':') || s.contains('|') {
                    Some(s.to_string())
                } else {
                    Some(format!("task: {} | query: ", s))
                }
            }
            Task::Raw => None,
        }
    }

    /// Format a text string with this task's prompt prefix.
    pub fn format_text(&self, text: &str) -> String {
        match self.prompt_prefix() {
            Some(prefix) => format!("{}{}", prefix, text),
            None => text.to_string(),
        }
    }

    /// Construct a Task from a name string (case-insensitive) and an optional document title.
    pub fn from_name_and_title(name: &'a str, title: Option<&'a str>) -> Self {
        match name.to_lowercase().replace(['-', '_', ' '], "").as_str() {
            "searchquery" | "retrieval" | "retrievalquery" | "query" => Task::SearchQuery,
            "document" | "retrievaldocument" => Task::Document { title },
            "coderetrieval" | "code" => Task::CodeRetrieval,
            "classification" | "multilabelclassification" => Task::Classification,
            "clustering" => Task::Clustering,
            "questionanswering" | "qa" => Task::QuestionAnswering,
            "factchecking" => Task::FactChecking,
            "sentencesimilarity" | "sts" | "pairclassification" | "summarization" => Task::SentenceSimilarity,
            "raw" | "none" => Task::Raw,
            _ => Task::Custom(name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_prompt_formatting() {
        assert_eq!(
            Task::SearchQuery.format_text("quantum physics"),
            "task: search result | query: quantum physics"
        );
        assert_eq!(
            Task::Document { title: None }.format_text("content body"),
            "title: none | text: content body"
        );
        assert_eq!(
            Task::Document { title: Some("Physics 101") }.format_text("content body"),
            "title: Physics 101 | text: content body"
        );
        assert_eq!(
            Task::CodeRetrieval.format_text("fn test()"),
            "task: code retrieval | query: fn test()"
        );
        assert_eq!(
            Task::Classification.format_text("sentiment review"),
            "task: classification | query: sentiment review"
        );
        assert_eq!(
            Task::Raw.format_text("unprompted text"),
            "unprompted text"
        );
    }

    #[test]
    fn test_task_from_name_and_title() {
        assert_eq!(
            Task::from_name_and_title("SearchQuery", None),
            Task::SearchQuery
        );
        assert_eq!(
            Task::from_name_and_title("document", Some("Doc A")),
            Task::Document { title: Some("Doc A") }
        );
        assert_eq!(
            Task::from_name_and_title("code_retrieval", None),
            Task::CodeRetrieval
        );
        assert_eq!(
            Task::from_name_and_title("qa", None),
            Task::QuestionAnswering
        );
    }
}
