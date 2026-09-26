"""Create audit and document status tables."""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "query_audit_logs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("document_id", sa.String(length=128), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("route", sa.String(length=32)),
        sa.Column("routing_path", sa.JSON(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("final_output", sa.JSON()),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_query_audit_logs_document_id", "query_audit_logs", ["document_id"])
    op.create_index("ix_query_audit_logs_status", "query_audit_logs", ["status"])
    op.create_index("ix_query_audit_logs_created_at", "query_audit_logs", ["created_at"])
    op.create_table(
        "document_status",
        sa.Column("document_id", sa.String(length=128), primary_key=True),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("chunk_count", sa.Integer()),
        sa.Column("error", sa.Text()),
    )
    op.create_index("ix_document_status_status", "document_status", ["status"])


def downgrade() -> None:
    op.drop_index("ix_document_status_status", table_name="document_status")
    op.drop_table("document_status")
    op.drop_index("ix_query_audit_logs_created_at", table_name="query_audit_logs")
    op.drop_index("ix_query_audit_logs_status", table_name="query_audit_logs")
    op.drop_index("ix_query_audit_logs_document_id", table_name="query_audit_logs")
    op.drop_table("query_audit_logs")